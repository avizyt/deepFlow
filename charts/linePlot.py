import plotly.express as px
import numpy as np

t = np.linspace(0, 2*np.pi, 100)

fig = px.line(x=t, y=np.cos(t)**2, labels={'x': 't', 'y': 'cos(t)'})
fig.show()