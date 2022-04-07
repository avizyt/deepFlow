import plotly.graph_objects as go
import plotly.express as px

import plotly.figure_factory as ff
import plotly.tools as pt
import plotly.subplots as sp




import numpy as np


t = np.linspace(0, 2*np.pi, 100)
#
# fig = px.line(x=t, y=np.cos(t)**2, labels={'x': 't', 'y': 'cos(t)'})

fig = sp.make_subplots(rows=2, shared_xaxes=True)

# [ (1,1) x1,y1 ]
# [ (2,1) x2,y2 ]

fig['data'] += [px.line(x=t, y=np.cos(t)**2)]
fig['data'] += [px.line(x=t, y=np.sin(t)**2)]
fig.show()
