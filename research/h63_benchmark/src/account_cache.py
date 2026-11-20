"""Mechanical speed adapter for the frozen engine; no allocation/account rule edits.

Cache invariant quote preparation, omit verbose per-name JSON, and emit only the
position columns used by the unchanged stop policy. All numerical ledger fields
must reproduce the frozen engine, including on randomized target paths.
"""
from pathlib import Path
import hashlib
import types


def build(portfolio):
    original = portfolio.exits.engine()
    source = Path(original.__file__).read_text(encoding='utf-8-sig')
    edits = []

    def replace(before, after):
        nonlocal source
        if source.count(before) != 1:
            raise ValueError('Frozen account source changed; optimization must be reviewed')
        source = source.replace(before, after)
        edits.append(dict(before=before, after=after))

    start = source.index('        bounded={}\n')
    end = source.index('    position_limit=', start)
    block = source[start:end]
    replace(block, "        if '_mc_bounded' not in globals():\n" +
            ''.join('    '+line if line.strip() else line for line in block.splitlines(True)) +
            "            globals()['_mc_bounded']=bounded\n        prices=_mc_bounded\n")
    replace('    quotes=DatedQuoteCache(data,names,clock)',
            "    if '_mc_quotes' not in globals():globals()['_mc_quotes']=DatedQuoteCache(data,names,clock)\n    quotes=_mc_quotes")
    begin=source.index('    risk=None if risk_betas is None else risk_betas.copy()')
    end=source.index('    e=_event_table(',begin)
    block=source[begin:end]
    replace(block,"    if '_mc_risk' not in globals():\n"+
        ''.join('    '+line if line.strip() else line for line in block.splitlines(True))+
        "        globals()['_mc_risk']=(risk,eligibility_frame)\n    risk,eligibility_frame=_mc_risk\n")
    begin=source.index('    # Prior marks/factors are read only through the initial decision close.')
    end=source.index('    previous_close=last.copy();',begin)
    block=source[begin:end]
    replace(block,"    if '_mc_initial' not in globals():\n"+
        ''.join('    '+line if line.strip() else line for line in block.splitlines(True))+
        "        globals()['_mc_initial']=(last.copy(),segment.copy(),factor.copy())\n    last,segment,factor=(x.copy() for x in _mc_initial)\n")
    replace("        broken=np.array([shares[j]!=0 and segment[j] is not None and pd.notna(seg[j]) and seg[j]!=segment[j] for j in range(n)])",
        "        broken=(shares!=0)&(segment!=None)&pd.notna(seg)&(seg!=segment)")
    begin = source.index('        if record_positions:\n            for j,t in enumerate(names):position_rows.append')
    end = source.index('        for j,t in enumerate(names):\n            if close_ok[j]', begin)
    replace(source[begin:end], '''        if record_positions:
            if position_sink is None:raise ValueError('Optimized mode requires a position sink')
            if hasattr(position_sink,'consume_arrays'):
                position_sink.consume_arrays(date,shares,closing_values,close_ok,unpriced_close,broken)
            else:
                position_sink(pd.DataFrame(dict(Date=date,ticker=names,closing_shares=shares.copy(),
                    closing_value=closing_values.copy(),close_quote_valid=close_ok.copy(),
                    mark_provisional=unpriced_close.copy(),segment_break=broken.copy())))
            streamed_position_rows+=n
''')
    begin=source.index('        for j,t in enumerate(names):\n            if close_ok[j]')
    end=source.index('        previous_nav=nav;',begin)
    replace(source[begin:end],'''        update=close_ok|(open_ok&~broken)
        last[update]=closing[update];segment[update]=seg[update]
        raw=quotes.row('raw_close',date)
        update=close_ok&np.isfinite(raw)&(raw>0)
        factor[update]=quote_close[update]/raw[update]
''')
    engine=types.ModuleType('v39_cached_account');engine.__file__=original.__file__
    exec(compile(source, original.__file__+' [cached]', 'exec'),engine.__dict__)
    original_inputs=engine._inputs
    cached={}

    def inputs(prices,weights,start,end,cutoff):
        key=(tuple(weights.columns),str(start),str(end),str(cutoff),id(prices))
        if not cached:
            cached['key']=key;cached['data']=original_inputs(prices,weights,start,end,cutoff)
        if cached['key']!=key:raise ValueError('Use a new cached engine for a different input dataset')
        names,_,clock,dates,data=cached['data']
        w=weights.loc[(weights.index<cutoff)&(weights.index<engine.pd.Timestamp(end))].copy()
        if (not w.index.is_unique or not w.index.is_monotonic_increasing or
            not w.index.isin(clock).all() or not engine.np.isfinite(w.to_numpy()).all()):
            raise ValueError('Invalid desired targets')
        return names,w,clock,dates,data

    engine._inputs=inputs
    engine._json=lambda names,values: ''  # Diagnostics only; decisions use numerical arrays.
    return engine,dict(source_sha256=hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
                       compiled_sha256=hashlib.sha256(source.encode()).hexdigest(),edits=edits)
