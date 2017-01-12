from matmodlab2.core.environ import environ

def create_figure(**kwargs):
    bokeh = kwargs.pop('bokeh', False)
    if bokeh:
        from bokeh.plotting import figure
        TOOLS = ('resize,crosshair,pan,wheel_zoom,box_zoom,'
                 'reset,box_select,lasso_select')
        TOOLS = 'resize,pan,wheel_zoom,box_zoom,reset,save'
        return figure(tools=TOOLS, **kwargs)
    else:
        from matplotlib import pylab
        return pylab.subplot()
