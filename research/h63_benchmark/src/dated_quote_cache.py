"""Exact storage adapter for already bounded/validated account input frames.

No interpolation, new eligibility rule or price transformation. A request gets
only its literal date row; matrices are immutable to prevent accidental writes.
"""
import numpy as np
import pandas as pd

class DatedQuoteCache:
    def __init__(self,data,names,clock):
        self.clock=pd.DatetimeIndex(clock);assert self.clock.is_unique
        self.positions={date:i for i,date in enumerate(self.clock)};self.fields={}
        specs=[('adj_open',float,np.nan),('adj_close',float,np.nan),('__open_ok',bool,False),
               ('__close_ok',bool,False),('tr_segment_id',object,np.nan),
               ('distribution_reinvestment_turnover',float,0.),('raw_close',float,np.nan)]
        for field,dtype,default in specs:
            columns=[]
            for name in names:
                if field in data[name]:values=data[name][field].reindex(self.clock).to_numpy(dtype=dtype)
                else:values=np.full(len(self.clock),default,dtype=dtype)
                columns.append(values)
            matrix=np.column_stack(columns);matrix.flags.writeable=False;self.fields[field]=matrix

    def row(self,field,date):return self.fields[field][self.positions[pd.Timestamp(date)]]
