import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as py
from plotly.subplots import make_subplots
py.init_notebook_mode(connected=True)

__name__ = "rsdata_charts"

def pie_chart(datasets, labels, datasets_names, chart_title, height, width, colors, export_name):  
        if len(datasets) == len(labels) and len(datasets_names) == len(labels):
            
            with open(export_name+".html", 'a', encoding='utf-8') as f:
                if len(datasets) <= 4:
                    if len(datasets) == 1:
                        specification = specs=[[{"type": "domain"}]]
                    elif len(datasets) == 2:
                        specification = specs=[[{"type": "domain"}, {"type": "domain"}]]
                    elif len(datasets) == 3:
                        specification = specs=[[{"type": "domain"}, {"type": "domain"}, {"type": "domain"}]]
                    else:
                        specification = specs=[[{"type": "domain"}, {"type": "domain"}, {"type": "domain"}, {"type": "domain"}]]
                    
                    fig = make_subplots(rows=1, cols=len(datasets),
                            specs=specification,
                            subplot_titles=(datasets_names))
                    
                    for i in range(len(datasets)):
                        data = datasets[i]
                        label_col = labels[i]
                        values = []
                        
                        for label in list(data[label_col].unique()):
                            values.append(data.query(label_col+" == '"+label+"'")[label_col].count())
                    
                        fig.add_trace(
                            go.Pie(labels=list(data[label_col].unique()),
                                     values=values,
                                     sort=False
                                     ),
                            row=1, col=i+1
                        )
                        
                    fig.update_traces(marker=dict(colors=colors))

                    fig.update_traces(textinfo='label+percent+value')
                    
                    fig.update_layout(height=height, width=width, title_text=chart_title, showlegend = False) #legend_orientation="h"
                    
                    fig.write_image(export_name+".jpeg")
                    
                    f.write(fig.to_html())
                
                
                else:
                    print("Not implemented yet")
        
        else:
            print("Parameters provided for datasets have different sizes")
            

def line_chart(datasets_names, traces, labels, legends, modes, colors, chart_title, x_title, y_title, height, width, legend_orientation="h", guidance="vertical", export_name="line_chart"): 
    #Checar se nomes dos datasets tem mesmo tamanho dos datasets 
    #if len(traces) == len(labels) and len(datasets_names) == len(labels): #aqui vao ser os únicos (os múltiplos tem adições próprias)
    if True:
        if len(datasets_names) < 1:
            print("No dataset informed for chart composition")
        
        if len(datasets_names) == 1:
            with open(export_name+".html", 'a', encoding='utf-8') as f:
                fig = make_subplots(rows=1, cols=len(datasets_names),
                                shared_yaxes=True,
                                subplot_titles=(datasets_names))
                
                #Adding Mean - loop com classes 
                for i in range(len(traces)):
                    if modes == 'markers+lines':
                        fig.add_trace(
                            go.Scatter(x = labels[i],
                                       y = traces[i],
                                       name = legends[i],
                                       mode = 'markers+lines',
                                       marker =  {'color' : colors[i],
                                                             'line' : {'width': 1,
                                                                       'color': colors[i]}},
                                       opacity=1),
                                row=1, col=1
                            )
                    if modes == 'lines':
                        fig.add_trace(
                            go.Scatter(x = labels[i],
                                       y = traces[i],
                                       name = legends[i],
                                       mode = 'lines',
                                       marker =  {'color' : colors[i],
                                                             'line' : {'width': 1,
                                                                       'color': colors[i]}},
                                       opacity=1),
                                row=1, col=1
                            )
                        
                    elif modes == 'dash':
                        fig.add_trace(
                            go.Scatter(x = labels[i],
                                       y = traces[i],
                                       name = legends[i],
                                       mode = 'lines',
                                       line =  {'color' : colors[i],
                                                'dash' : 'dash'},
                                       opacity=1),
                                row=1, col=1
                            )
                        
                    elif modes == 'dot':
                        fig.add_trace(
                            go.Scatter(x = labels[i],
                                       y = traces[i],
                                       name = legends[i],
                                       mode = 'lines',
                                       line =  {'color' : colors[i],
                                                'dash' : 'dot'},
                                       opacity=1),
                                row=1, col=1
                            )
                    
                fig.update_xaxes(title_text=x_title)
                fig.update_yaxes(title_text=y_title)
                fig.update_layout(height=height, width=width, title_text=chart_title, template = 'plotly_white')
                fig.update_layout(legend_orientation=legend_orientation)#, legend=dict(x=0.0, y=-0.4)
                fig.write_image(export_name+".jpeg")
                f.write(fig.to_html())
                
        elif guidance == "horizontal": 
            if len(datasets_names) <= 4:
                with open(export_name+".html", 'a', encoding='utf-8') as f:
                    fig = make_subplots(rows=1, cols=len(datasets_names),
                                    shared_yaxes=True,
                                    subplot_titles=(datasets_names))
                    for i in range(len(datasets_names)):
                        trace = traces[i]
                        label = labels[i]
                        legend = legends[i]
                        color = colors[i]
                        mode = modes[i]
                        

                        for j in range(len(trace)):
                            if mode[j] == 'markers+lines':
                                fig.add_trace(
                                    go.Scatter(x = label[j],
                                               y = trace[j],
                                               name = legend[j],
                                               mode = 'markers+lines',
                                               marker =  {'color' : color[j],
                                                                     'line' : {'width': 1,
                                                                               'color': color[j]}},
                                               opacity=1),
                                        row=1, col=(i+1)
                                    )
                            elif mode[j] == 'lines':
                                fig.add_trace(
                                    go.Scatter(x = label[j],
                                               y = trace[j],
                                               name = legend[j],
                                               mode = 'lines',
                                               marker =  {'color' : color[j],
                                                                     'line' : {'width': 1,
                                                                               'color': color[j]}},
                                               opacity=1),
                                        row=1, col=(i+1)
                                    )

                            elif mode[j] == 'dash':
                                fig.add_trace(
                                    go.Scatter(x = label[j],
                                               y = trace[j],
                                               name = legend[j],
                                               mode = 'lines',
                                               line =  {'color' : color[j],
                                                        'dash' : 'dash'},
                                               opacity=1),
                                        row=1, col=(i+1)
                                    )

                            elif mode[j] == 'dot':
                                fig.add_trace(
                                    go.Scatter(x = label[j],
                                               y = trace[j],
                                               name = legend[j],
                                               mode = 'lines',
                                               line =  {'color' : color[j],
                                                        'dash' : 'dot'},
                                               opacity=1),
                                        row=1, col=(i+1)
                                    )

                    fig.update_xaxes(title_text=x_title)
                    fig.update_yaxes(title_text=y_title)
                    fig.update_layout(height=height, width=width, title_text=chart_title, template = 'plotly_white')
                    fig.update_layout(legend_orientation=legend_orientation) #, legend=dict(x=0.0, y=-0.4)
                    fig.write_image(export_name+".jpeg")
                    f.write(fig.to_html())
            else:
                print("Maximum 4 data sets exceeded. Try to spread your data over more than one chart.")
                
        elif guidance == "vertical": 
            with open(export_name+".html", 'a', encoding='utf-8') as f:
                    fig = make_subplots(rows=len(datasets_names), cols=1,
                                    shared_xaxes=True,
                                    subplot_titles=(datasets_names))
                    
                    for i in range(len(datasets_names)):
                        trace = traces[i]
                        label = labels[i]
                        legend = legends[i]
                        color = colors[i]
                        mode = modes[i]


                        for j in range(len(trace)):
                            if mode[j] == 'markers+lines':
                                fig.add_trace(
                                    go.Scatter(x = label[j],
                                               y = trace[j],
                                               name = legend[j],
                                               mode = 'markers+lines',
                                               marker =  {'color' : color[j],
                                                                     'line' : {'width': 1,
                                                                               'color': color[j]}},
                                               opacity=1),
                                        col=1, row=(i+1)
                                    )
                            
                            elif mode[j] == 'lines':
                                fig.add_trace(
                                    go.Scatter(x = label[j],
                                               y = trace[j],
                                               name = legend[j],
                                               mode = 'markers+lines',
                                               marker =  {'color' : color[j],
                                                                     'line' : {'width': 1,
                                                                               'color': color[j]}},
                                               opacity=1),
                                        col=1, row=(i+1)
                                    )

                            elif mode[j] == 'dash':
                                fig.add_trace(
                                    go.Scatter(x = label[j],
                                               y = trace[j],
                                               name = legend[j],
                                               mode = 'lines',
                                               line =  {'color' : color[j],
                                                        'dash' : 'dash'},
                                               opacity=1),
                                        col=1, row=(i+1)
                                    )

                            elif mode[j] == 'dot':
                                fig.add_trace(
                                    go.Scatter(x = label[j],
                                               y = trace[j],
                                               name = legend[j],
                                               mode = 'lines',
                                               line =  {'color' : color[j],
                                                        'dash' : 'dot'},
                                               opacity=1),
                                        col=1, row=(i+1)
                                    )
                                
                    fig.update_xaxes(title_text=x_title)
                    fig.update_yaxes(title_text=y_title)
                    fig.update_layout(height=height, width=width, title_text=chart_title, template = 'plotly_white')
                    fig.update_layout(legend_orientation=legend_orientation)#legend=dict(x=0.0, y=-0.4)
                    fig.write_image(export_name+".jpeg")
                    f.write(fig.to_html())
        
        else:
                print("Invalid value for 'guidance' parameter")
    
    else:
        print("Parameters provided for datasets have different sizes")
        

def bar_chart(datasets_names, traces, labels, color, line_color, chart_title, x_title, y_title, height, width, orientation, guidance="vertical", export_name="bar_chart"): 
    if len(datasets_names) == 1:
        with open(export_name+".html", 'a', encoding='utf-8') as f:
            fig = make_subplots(rows=1, cols=1) 

            fig.add_trace(go.Figure(go.Bar(
                    x=traces,
                    y=labels,
                    marker=dict(
                        color=color,
                        line=dict(
                            color=line_color,
                            width=1),
                        ),
                    orientation=legend_orientation)),
                    row=1, col=1
                )
            
            fig.update_xaxes(title_text=x_title)
            fig.update_yaxes(title_text=y_title)
            fig.update_layout(height=height, width=width, title_text=datasets_names[0], template = 'plotly_white')
            fig.update_layout(showlegend = False)#, legend=dict(x=0.0, y=-0.4)
            fig.write_image(export_name+".jpeg")
            f.write(fig.to_html())

    elif guidance == "horizontal": 
        if len(datasets_names) <= 4:
            with open(export_name+".html", 'a', encoding='utf-8') as f:
                fig = make_subplots(rows=1, cols=len(datasets_names),
                                shared_yaxes=True,
                                shared_xaxes=True,
                                subplot_titles=(datasets_names))
                
                for i in range(len(datasets_names)):
                    fig.add_trace(go.Bar(
                            x=traces[i],
                            y=labels[i],
                            marker=dict(
                                color=color,
                                line=dict(
                                    color=line_color,
                                    width=1),
                                ),
                            orientation=orientation),
                            row=1, col=(i+1)
                        )
                
                fig.update_xaxes(title_text=x_title)
                fig.update_yaxes(title_text=y_title)
                fig.update_layout(height=height, width=width, title_text=chart_title, template = 'plotly_white')
                fig.update_layout(showlegend = False)#, legend=dict(x=0.0, y=-0.4)
                fig.write_image(export_name+".jpeg")
                f.write(fig.to_html())

        else:
            print("Maximum 4 data sets exceeded. Try to spread your data over more than one chart.")
    else: 
        print("Guidance = vertical not implented yet")




def stacked_bar_chart(datasets_names, x, y, names, colors, chart_title, x_title, y_title, height, width, labels_group=" ", orientation="v", guidance="vertical", export_name="stacked_bar_chart"):
    if len(datasets_names) == 1:
        with open(export_name+".html", 'a', encoding='utf-8') as f:
            fig = make_subplots(rows=1, cols=1) 

            for i in range(len(x)):
                fig.add_trace(go.Bar(
                    x=x[i],
                    y=y[i],
                    name=names[i],
                    marker_color=colors[i]),
                    row=1, col=1
                )
            
            fig.update_xaxes(title_text=x_title)
            fig.update_yaxes(title_text=y_title)
            fig.update_layout(height=height, width=width, title_text=datasets_names[0], template = 'plotly_white')
            fig.write_image(export_name+".jpeg")
            f.write(fig.to_html())

    elif guidance == "horizontal": 
        if len(datasets_names) <= 5:
            with open(export_name+".html", 'a', encoding='utf-8') as f:
                fig = make_subplots(rows=1, cols=len(datasets_names),
                                shared_yaxes=True,
                                shared_xaxes=True,
                                subplot_titles=(datasets_names))
                
                for i in range(len(y)):
                    z = y[i]
                    for j in range(len(x)):
                        fig.add_trace(go.Bar(
                            x=x[j],
                            y=z[j],
                            legendgroup=i,
                            legendgrouptitle_text=labels_group[i],
                            name=names[j],
                            marker_color=colors[j]),
                            row=1, col=i+1
                        )
               
                fig.update_xaxes(title_text=x_title)
                fig.update_yaxes(title_text=y_title)
                fig.update_layout(barmode = 'stack')           
                fig.update_layout(height=height, width=width, title_text=chart_title, template = 'plotly_white')
                fig.write_image(export_name+".jpeg")
                f.write(fig.to_html())

        else:
            print("Maximum 5 data sets exceeded. Try to spread your data over more than one chart.")
    else: 
        print("Guidance = vertical not implented yet")
        

        
        
def scatter_chart_und(datasets_names, traces, y, labels, labels_group, colors, chart_title, x_title, y_title, height, width, legend_orientation="h", guidance="vertical", export_name="scatter_chart"): 
    if guidance == "horizontal": 
        if len(datasets_names) <= 5:
            with open(export_name+".html", 'a', encoding='utf-8') as f:
                fig = make_subplots(rows=1, cols=len(datasets_names),
                                shared_yaxes=True,
                                shared_xaxes=True,
                                subplot_titles=(datasets_names))
                
                for i in range(len(datasets_names)):
                    trace = traces[i]
                    label = labels[i]
                    color = colors[i]
                    
                    for j in range(len(trace)):
                        fig.add_trace(
                            go.Scatter(x = trace[j].index,
                                       y = trace[j][y],
                                       name = label[j],
                                       legendgroup=i,
                                       legendgrouptitle_text=labels_group[i],
                                       mode = 'markers',
                                       marker =  {'color' : color[j]},
                                       opacity=1),
                                col=(i+1), row=1
                            )
                        
                fig.update_xaxes(title_text=x_title)
                fig.update_yaxes(title_text=y_title)
                fig.update_layout(height=height, width=width, title_text=chart_title, template = 'plotly_white')
                fig.update_layout(legend_orientation=legend_orientation)#, legend=dict(x=0.0, y=-0.4)
                fig.write_image(export_name+".jpeg")
                f.write(fig.to_html())

        else:
            print("Maximum 4 data sets exceeded. Try to spread your data over more than one chart.")

def scatter_chart(datasets_names, traces, x, y, labels, labels_group, colors, chart_title, x_title, y_title, height, width, legend_orientation="h", guidance="vertical", export_name="scatter_chart"): 
    if len(datasets_names) == 1:
        with open(export_name+".html", 'a', encoding='utf-8') as f:
            fig = make_subplots(rows=1, cols=1) 

            for i in range(len(traces)):
                fig.add_trace(go.Scatter(x = traces[i][x],
                           y = traces[i][y],
                           name = labels[i],
                           mode = 'markers',
                           marker =  {'color' : colors[i]},
                           opacity=1),
                    row=1, col=1
                )
            
            fig.update_xaxes(title_text=x_title)
            fig.update_yaxes(title_text=y_title)
            fig.update_layout(height=height, width=width, title_text=datasets_names[0], template = 'plotly_white')
            fig.update_layout(legend_orientation=legend_orientation)#, legend=dict(x=0.0, y=-0.4)
            fig.write_image(export_name+".jpeg")
            f.write(fig.to_html())

    elif guidance == "horizontal": 
        if len(datasets_names) <= 4:
            with open(export_name+".html", 'a', encoding='utf-8') as f:
                fig = make_subplots(rows=1, cols=len(datasets_names),
                                shared_yaxes=True,
                                shared_xaxes=True,
                                subplot_titles=(datasets_names))
                
                for i in range(len(datasets_names)):
                    trace = traces[i]
                    label = labels[i]
                    color = colors[i]
                    
                    for j in range(len(trace)):
                        fig.add_trace(
                            go.Scatter(x = trace[j][x],
                                       y = trace[j][y],
                                       name = label[j],
                                       legendgroup=i,
                                       legendgrouptitle_text=labels_group[i],
                                       mode = 'markers',
                                       marker =  {'color' : color[j]},
                                       opacity=1),
                                col=(i+1), row=1
                            )
                        
                fig.update_xaxes(title_text=x_title)
                fig.update_yaxes(title_text=y_title)
                fig.update_layout(height=height, width=width, title_text=chart_title, template = 'plotly_white')
                fig.update_layout(legend_orientation=legend_orientation)#, legend=dict(x=0.0, y=-0.4)
                fig.write_image(export_name+".jpeg")
                f.write(fig.to_html())

        else:
            print("Maximum 4 data sets exceeded. Try to spread your data over more than one chart.")

            

def boxplot_chart(datasets_names, traces, labels, colors, chart_title, x_title, y_title, height, width, guidance="vertical", export_name="boxplot_chart"): 
    if len(datasets_names) == 1:
        with open(export_name+".html", 'a', encoding='utf-8') as f:
            fig = make_subplots(rows=1, cols=1) 

            for i in range(len(traces)):
                fig.add_trace(go.Box(y=traces[i],
                                     name=labels[i],
                                     marker = {'color': colors[i]}), 
                          row=1, col=1)

            fig.update_layout(height=height, width=width, title_text=datasets_names[0], template = 'plotly_white')
            fig.write_image(export_name+".jpeg")
            f.write(fig.to_html())
    
    elif guidance == "horizontal":
        if len(datasets_names) <= 4:
            with open(export_name+".html", 'a', encoding='utf-8') as f:
                fig = make_subplots(rows=1, cols=len(datasets_names),
                                    shared_yaxes=True,
                                    shared_xaxes=True,
                                    subplot_titles=(datasets_names))

                for i in range(len(datasets_names)):
                    trace = traces[i]
                    label = labels[i]
                    color = colors[i]
                    
                    for j in range(len(trace)):
                        fig.add_trace(go.Box(y=trace[j],
                                         name=label[j],
                                         marker = {'color': color[j]}), 
                              row=1, col=(i+1))

                fig.update_layout(height=height, width=width, title_text=chart_title, showlegend = False, template = 'plotly_white')
                fig.write_image(export_name+".jpeg")
                f.write(fig.to_html())
                    
            
        else:
            print("Maximum 4 data sets exceeded. Try to spread your data over more than one chart.")
   
        
def heatmap_chart(datasets_names, datasets, x_labels, y_labels, colorscale, chart_title, height, width,  guidance="vertical", export_name="heatmap_chart"):
    if len(datasets_names) == 1:
        with open(export_name+".html", 'a', encoding='utf-8') as f:
            fig = make_subplots(rows=1, cols=1) 

            fig.add_trace(go.Heatmap(x=x_labels, 
                                     y=y_labels,
                                     z=datasets, 
                                     texttemplate= "%{z}",
                                     coloraxis="coloraxis"), 
                          row=1, col=1)

            fig.update_layout(coloraxis=dict(colorscale=colorscale))
            fig.update_layout(height=height, width=width, title_text=datasets_names[0], template = 'plotly_white')
            fig.write_image(export_name+".jpeg")
            f.write(fig.to_html())
            
            
    elif guidance == "horizontal":
        if len(datasets_names) <= 4:
            with open(export_name+".html", 'a', encoding='utf-8') as f:
                fig = make_subplots(rows=1, cols=len(datasets_names), subplot_titles=datasets_names, horizontal_spacing=0.1) 
                
                for i in range(len(datasets_names)):                 
                    fig.add_trace(go.Heatmap(x=x_labels[i], 
                                             y=y_labels[i],
                                             z=datasets[i], 
                                             texttemplate= "%{z}",
                                             coloraxis="coloraxis"), 
                                  row=1, col=i+1)

                fig.update_layout(coloraxis=dict(colorscale=colorscale))
                fig.update_layout(height=height, width=width, title_text=chart_title, template = 'plotly_white')
                fig.write_image(export_name+".jpeg")
                f.write(fig.to_html())
        else:
            print("Maximum 4 data sets exceeded. Try to spread your data over more than one chart.")
            
            
    elif guidance == "vertical":
        with open(export_name+".html", 'a', encoding='utf-8') as f:
            fig = make_subplots(rows=len(datasets_names), cols=1, subplot_titles=datasets_names, vertical_spacing=0.08) 
                
            for i in range(len(datasets_names)):                 
                fig.add_trace(go.Heatmap(x=x_labels[i], 
                                         y=y_labels[i],
                                         z=datasets[i], 
                                         texttemplate= "%{z}",
                                         coloraxis="coloraxis"), 
                              row=i+1, col=1)

            fig.update_layout(coloraxis=dict(colorscale=colorscale))
            fig.update_layout(height=height, width=width, title_text=chart_title, template = 'plotly_white')
            fig.write_image(export_name+".jpeg")
            f.write(fig.to_html())
                
        
def stackedbars_chart(datasets_names, traces, bands, n_bins, labels, labels_group, colors, chart_title, x_title, y_title, height, width, export_name):   
    if True: #restrição futura de tamanhos entre datasets, labels e etc
        if len(datasets_names) < 1:
            print("No dataset informed for chart composition")
        
        elif len(datasets_names) <= 4:
            with open(export_name+".html", 'a', encoding='utf-8') as f:
                #print(trace, band, label, color)
                fig = make_subplots(rows=1, cols=len(datasets_names),
                                shared_yaxes=True,
                                shared_xaxes=True,
                                subplot_titles=(datasets_names))
                
                for i in range(len(datasets_names)): 
                    trace = traces[i]
                    band = bands[i]
                    label = labels[i]
                    color = colors[i]
                    
                    for j in range(len(trace)):

                        fig.add_trace(go.Histogram(
                            x=trace[j][band],
                            histnorm='percent',
                            name=label[j], 
                            legendgroup=i,
                            nbinsx=n_bins,
                            legendgrouptitle_text=labels_group[i],
                            marker_color=color[j],
                            opacity=0.9), 
                            row=1, col=i+1)
  
                fig.update_xaxes(title_text=x_title)
                fig.update_yaxes(title_text=y_title)
                fig.update_layout(height=height, width=width, title_text=chart_title, template = 'plotly_white')    
                fig.write_image(export_name+".jpeg")
                f.write(fig.to_html())
        else:
                print("Not implemented yet")
    
    else:
        print("Parameters provided for datasets have different sizes")
        
def map_nn(date, ground_truth, classified_data, path, caminho, height, width): #caminho, 
    #Ground truth spacial info
    #English
    mapa = []
    color = []
    data = ground_truth.query("Path == '"+date+"'")
    
    for i in range(len(set(list(data['Line']))) - 1):
        map_line = []
        color_line = []
        data_line = data.loc[data['Line'] == i]
        for j in range(len(set(list(data['Column']))) - 1):
            cell = data_line.loc[data_line['Column'] == j]['Label'].values
            if len(cell) > 0:
                map_line.append(cell[0])
                #if cell[0] == 'Sand' or cell[0] == 'Coast':
                #    color_line.append(-20)
                if cell[0] == 'Water':
                    color_line.append(-10)
                elif cell[0] == 'Plastic':
                    color_line.append(10)
                #elif cell[0] == 'Wood':
                #    color_line.append(20)
            else:
                map_line.append("XXXXXX")
                color_line.append(0)

        mapa.append(map_line)
        color.append(color_line)

    dat = date.replace('/','_')
    export_name = path+dat+'_groundtruth'
    
    with open(export_name+".html", 'a', encoding='utf-8') as f:
        fig = go.Figure(data=go.Heatmap(
                    z=color,
                    text=mapa,
                    texttemplate="%{text}",
                    textfont={"size":11}))

        fig.update_layout(height=height, width=width)
        fig.write_image(export_name+".jpeg")
                    
        f.write(fig.to_html())
    
    #Portuguese
    mapa = []
    color = []
    data = ground_truth.query("Path == '"+date+"'")
    
    for i in range(len(set(list(data['Line']))) - 1):
        map_line = []
        color_line = []
        data_line = data.loc[data['Line'] == i]
        for j in range(len(set(list(data['Column']))) - 1):
            cell = data_line.loc[data_line['Column'] == j]['Classe'].values
            if len(cell) > 0:
                map_line.append(cell[0])
                #if cell[0] == 'Areia' or cell[0] == 'Costa':
                #    color_line.append(-20)
                if cell[0] == 'Água':
                    color_line.append(-10)
                elif cell[0] == 'Plástico':
                    color_line.append(10)
                #elif cell[0] == 'Madeira':
                #    color_line.append(20)
            else:
                map_line.append("XXXXXX")
                color_line.append(0)

        mapa.append(map_line)
        color.append(color_line)

    dat = date.replace('/','_')
    export_name = caminho+dat+'_verdadecampo'
    
    with open(export_name+".html", 'a', encoding='utf-8') as f:
        fig = go.Figure(data=go.Heatmap(
                    z=color,
                    text=mapa,
                    texttemplate="%{text}",
                    textfont={"size":11}))

        fig.update_layout(height=height, width=width)
        fig.write_image(export_name+".jpeg")
                    
        f.write(fig.to_html())
        

    '''
    #---------------------------
    #Classification spacial info
    #English
    mapa = []
    color = []
    data = classified_data.query("Path == '"+date+"'")

    for i in range(len(set(list(data['Line']))) - 1):
        map_line = []
        color_line = []
        data_line = data.loc[data['Line'] == i]
        for j in range(len(set(list(data['Column']))) - 1):
            cell = data_line.loc[data_line['Column'] == j]['Cluster'].values
            if len(cell) > 0:
                map_line.append(cell[0])
                #if cell[0] == 'Sand' or cell[0] == 'Coast':
                #    color_line.append(-20)
                if cell[0] == 'Water':
                    color_line.append(-10)
                elif cell[0] == 'Plastic':
                    color_line.append(10)
                #elif cell[0] == 'Wood':
                #    color_line.append(20)
            else:
                map_line.append("XXXXXX")
                color_line.append(0)

        mapa.append(map_line)
        color.append(color_line)

    dat = date.replace('/','_')
    export_name = path+dat+'_classified'
    
    with open(export_name+".html", 'a', encoding='utf-8') as f:
        fig = go.Figure(data=go.Heatmap(
                    z=color,
                    text=mapa,
                    texttemplate="%{text}",
                    textfont={"size":11}))
        
        fig.update_layout(height=height, width=width)
        fig.write_image(export_name+".jpeg")
                    
        f.write(fig.to_html())

    #Portuguese
    mapa = []
    color = []
    data = classified_data.query("Path == '"+date+"'")

    for i in range(len(set(list(data['Line']))) - 1):
        map_line = []
        color_line = []
        data_line = data.loc[data['Line'] == i]
        for j in range(len(set(list(data['Column']))) - 1):
            cell = data_line.loc[data_line['Column'] == j]['Cluster'].values
            if len(cell) > 0:
                map_line.append(cell[0])
                #if cell[0] == 'Areia' or cell[0] == 'Costa':
                #    color_line.append(-20)
                if cell[0] == 'Água':
                    color_line.append(-10)
                elif cell[0] == 'Plástico':
                    color_line.append(10)
                #elif cell[0] == 'Madeira':
                #    color_line.append(20)
            else:
                map_line.append("XXXXXX")
                color_line.append(0)

        mapa.append(map_line)
        color.append(color_line)

    dat = date.replace('/','_')
    export_name = caminho+dat+'_classificado'
    
    with open(export_name+".html", 'a', encoding='utf-8') as f:
        fig = go.Figure(data=go.Heatmap(
                    z=color,
                    text=mapa,
                    texttemplate="%{text}",
                    textfont={"size":11}))

        fig.update_layout(height=height, width=width)
        fig.write_image(export_name+".jpeg")
                    
        f.write(fig.to_html())
    
    '''