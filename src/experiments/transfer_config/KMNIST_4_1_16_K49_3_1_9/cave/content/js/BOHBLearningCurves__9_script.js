
  (function() {
    var fn = function() {
      Bokeh.safely(function() {
        (function(root) {
          function embed_document(root) {
            
          var docs_json = '{"13b633bc-7c0e-4990-9d78-5227b097850d":{"roots":{"references":[{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":20},"x":{"field":"times"},"y":{"field":"losses"}},"id":"1225","type":"CircleX"},{"attributes":{"filters":[{"id":"1187","type":"GroupFilter"},{"id":"1188","type":"GroupFilter"}],"source":{"id":"1139","type":"ColumnDataSource"}},"id":"1189","type":"CDSView"},{"attributes":{},"id":"1146","type":"LinearScale"},{"attributes":{"data_source":{"id":"1139","type":"ColumnDataSource"},"glyph":{"id":"1224","type":"CircleX"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"1225","type":"CircleX"},"selection_glyph":null,"view":{"id":"1222","type":"CDSView"}},"id":"1226","type":"GlyphRenderer"},{"attributes":{},"id":"1148","type":"LogScale"},{"attributes":{"column_name":"HB_iteration","group":"2"},"id":"1227","type":"GroupFilter"},{"attributes":{"fill_alpha":{"value":0.5},"fill_color":{"field":"colors","transform":{"id":"1140","type":"LinearColorMapper"}},"line_color":{"field":"colors"},"size":{"units":"screen","value":20},"x":{"field":"times"},"y":{"field":"losses"}},"id":"1191","type":"Circle"},{"attributes":{"plot":{"id":"1143","subtype":"Figure","type":"Plot"},"ticker":{"id":"1151","type":"BasicTicker"}},"id":"1154","type":"Grid"},{"attributes":{"column_name":"config_info","group":"model_based_pick=False"},"id":"1228","type":"GroupFilter"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":20},"x":{"field":"times"},"y":{"field":"losses"}},"id":"1192","type":"Circle"},{"attributes":{},"id":"1151","type":"BasicTicker"},{"attributes":{"filters":[{"id":"1227","type":"GroupFilter"},{"id":"1228","type":"GroupFilter"}],"source":{"id":"1139","type":"ColumnDataSource"}},"id":"1229","type":"CDSView"},{"attributes":{"data_source":{"id":"1139","type":"ColumnDataSource"},"glyph":{"id":"1191","type":"Circle"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"1192","type":"Circle"},"selection_glyph":null,"view":{"id":"1189","type":"CDSView"}},"id":"1193","type":"GlyphRenderer"},{"attributes":{"axis_label":"Time","formatter":{"id":"1249","type":"BasicTickFormatter"},"plot":{"id":"1143","subtype":"Figure","type":"Plot"},"ticker":{"id":"1151","type":"BasicTicker"}},"id":"1150","type":"LinearAxis"},{"attributes":{"axis_label":"Cost","formatter":{"id":"1251","type":"LogTickFormatter"},"plot":{"id":"1143","subtype":"Figure","type":"Plot"},"ticker":{"id":"1156","type":"LogTicker"}},"id":"1155","type":"LogAxis"},{"attributes":{"column_name":"HB_iteration","group":"1"},"id":"1194","type":"GroupFilter"},{"attributes":{"fill_alpha":{"value":0.5},"fill_color":{"field":"colors","transform":{"id":"1140","type":"LinearColorMapper"}},"line_color":{"field":"colors"},"size":{"units":"screen","value":20},"x":{"field":"times"},"y":{"field":"losses"}},"id":"1231","type":"Circle"},{"attributes":{"dimension":1,"plot":{"id":"1143","subtype":"Figure","type":"Plot"},"ticker":{"id":"1156","type":"LogTicker"}},"id":"1159","type":"Grid"},{"attributes":{"filters":[{"id":"1194","type":"GroupFilter"}],"source":{"id":"1138","type":"ColumnDataSource"}},"id":"1195","type":"CDSView"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":20},"x":{"field":"times"},"y":{"field":"losses"}},"id":"1232","type":"Circle"},{"attributes":{"num_minor_ticks":10},"id":"1156","type":"LogTicker"},{"attributes":{},"id":"1257","type":"Selection"},{"attributes":{"data_source":{"id":"1139","type":"ColumnDataSource"},"glyph":{"id":"1231","type":"Circle"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"1232","type":"Circle"},"selection_glyph":null,"view":{"id":"1229","type":"CDSView"}},"id":"1233","type":"GlyphRenderer"},{"attributes":{"line_alpha":{"value":0.5},"line_color":{"field":"colors","transform":{"id":"1140","type":"LinearColorMapper"}},"line_width":{"value":5},"xs":{"field":"times"},"ys":{"field":"losses"}},"id":"1197","type":"MultiLine"},{"attributes":{},"id":"1164","type":"ResetTool"},{"attributes":{"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"line_width":{"value":5},"xs":{"field":"times"},"ys":{"field":"losses"}},"id":"1198","type":"MultiLine"},{"attributes":{"overlay":{"id":"1169","type":"BoxAnnotation"}},"id":"1163","type":"BoxZoomTool"},{"attributes":{"data_source":{"id":"1138","type":"ColumnDataSource"},"glyph":{"id":"1197","type":"MultiLine"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"1198","type":"MultiLine"},"selection_glyph":null,"view":{"id":"1195","type":"CDSView"}},"id":"1199","type":"GlyphRenderer"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"1137","type":"HoverTool"},{"id":"1160","type":"SaveTool"},{"id":"1161","type":"PanTool"},{"id":"1162","type":"WheelZoomTool"},{"id":"1163","type":"BoxZoomTool"},{"id":"1164","type":"ResetTool"}]},"id":"1165","type":"Toolbar"},{"attributes":{"active":[0,1,2],"callback":{"id":"1234","type":"CustomJS"},"labels":["00","01","02"]},"id":"1235","type":"CheckboxButtonGroup"},{"attributes":{"column_name":"HB_iteration","group":"0"},"id":"1174","type":"GroupFilter"},{"attributes":{"column_name":"HB_iteration","group":"1"},"id":"1200","type":"GroupFilter"},{"attributes":{},"id":"1160","type":"SaveTool"},{"attributes":{"args":{"checkbox":{"id":"1235","type":"CheckboxButtonGroup"},"glyph_renderer0":{"id":"1179","type":"GlyphRenderer"},"glyph_renderer1":{"id":"1186","type":"GlyphRenderer"},"glyph_renderer2":{"id":"1193","type":"GlyphRenderer"},"glyph_renderer3":{"id":"1199","type":"GlyphRenderer"},"glyph_renderer4":{"id":"1206","type":"GlyphRenderer"},"glyph_renderer5":{"id":"1213","type":"GlyphRenderer"},"glyph_renderer6":{"id":"1219","type":"GlyphRenderer"},"glyph_renderer7":{"id":"1226","type":"GlyphRenderer"},"glyph_renderer8":{"id":"1233","type":"GlyphRenderer"}},"code":"var labels = [0, 1, 2]; checkbox.active = labels;len_labels = 3;glyph_renderers = [[glyph_renderer0,glyph_renderer1,glyph_renderer2],[glyph_renderer3,glyph_renderer4,glyph_renderer5],[glyph_renderer6,glyph_renderer7,glyph_renderer8]];\\n        for (i = 0; i &lt; len_labels; i++) {\\n            if (checkbox.active.includes(i)) {\\n                // console.log(&#x27;Setting to true: &#x27; + i + &#x27;(&#x27; + glyph_renderers[i].length + &#x27;)&#x27;)\\n                for (j = 0; j &lt; glyph_renderers[i].length; j++) {\\n                    glyph_renderers[i][j].visible = true;\\n                    // console.log(&#x27;Setting to true: &#x27; + i + &#x27; : &#x27; + j)\\n                }\\n            } else {\\n                // console.log(&#x27;Setting to false: &#x27; + i + &#x27;(&#x27; + glyph_renderers[i].length + &#x27;)&#x27;)\\n                for (j = 0; j &lt; glyph_renderers[i].length; j++) {\\n                    glyph_renderers[i][j].visible = false;\\n                    // console.log(&#x27;Setting to false: &#x27; + i + &#x27; : &#x27; + j)\\n                }\\n            }\\n        }\\n        "},"id":"1236","type":"CustomJS"},{"attributes":{"column_name":"config_info","group":"model_based_pick=True"},"id":"1201","type":"GroupFilter"},{"attributes":{},"id":"1161","type":"PanTool"},{"attributes":{"callback":{"id":"1236","type":"CustomJS"},"icon":null,"label":"All"},"id":"1237","type":"Button"},{"attributes":{"filters":[{"id":"1200","type":"GroupFilter"},{"id":"1201","type":"GroupFilter"}],"source":{"id":"1139","type":"ColumnDataSource"}},"id":"1202","type":"CDSView"},{"attributes":{},"id":"1162","type":"WheelZoomTool"},{"attributes":{"args":{"checkbox":{"id":"1235","type":"CheckboxButtonGroup"},"glyph_renderer0":{"id":"1179","type":"GlyphRenderer"},"glyph_renderer1":{"id":"1186","type":"GlyphRenderer"},"glyph_renderer2":{"id":"1193","type":"GlyphRenderer"},"glyph_renderer3":{"id":"1199","type":"GlyphRenderer"},"glyph_renderer4":{"id":"1206","type":"GlyphRenderer"},"glyph_renderer5":{"id":"1213","type":"GlyphRenderer"},"glyph_renderer6":{"id":"1219","type":"GlyphRenderer"},"glyph_renderer7":{"id":"1226","type":"GlyphRenderer"},"glyph_renderer8":{"id":"1233","type":"GlyphRenderer"}},"code":"var labels = []; checkbox.active = labels;len_labels = 3;glyph_renderers = [[glyph_renderer0,glyph_renderer1,glyph_renderer2],[glyph_renderer3,glyph_renderer4,glyph_renderer5],[glyph_renderer6,glyph_renderer7,glyph_renderer8]];\\n        for (i = 0; i &lt; len_labels; i++) {\\n            if (checkbox.active.includes(i)) {\\n                // console.log(&#x27;Setting to true: &#x27; + i + &#x27;(&#x27; + glyph_renderers[i].length + &#x27;)&#x27;)\\n                for (j = 0; j &lt; glyph_renderers[i].length; j++) {\\n                    glyph_renderers[i][j].visible = true;\\n                    // console.log(&#x27;Setting to true: &#x27; + i + &#x27; : &#x27; + j)\\n                }\\n            } else {\\n                // console.log(&#x27;Setting to false: &#x27; + i + &#x27;(&#x27; + glyph_renderers[i].length + &#x27;)&#x27;)\\n                for (j = 0; j &lt; glyph_renderers[i].length; j++) {\\n                    glyph_renderers[i][j].visible = false;\\n                    // console.log(&#x27;Setting to false: &#x27; + i + &#x27; : &#x27; + j)\\n                }\\n            }\\n        }\\n        "},"id":"1238","type":"CustomJS"},{"attributes":{},"id":"1256","type":"UnionRenderers"},{"attributes":{"args":{"glyph_renderer0":{"id":"1179","type":"GlyphRenderer"},"glyph_renderer1":{"id":"1186","type":"GlyphRenderer"},"glyph_renderer2":{"id":"1193","type":"GlyphRenderer"},"glyph_renderer3":{"id":"1199","type":"GlyphRenderer"},"glyph_renderer4":{"id":"1206","type":"GlyphRenderer"},"glyph_renderer5":{"id":"1213","type":"GlyphRenderer"},"glyph_renderer6":{"id":"1219","type":"GlyphRenderer"},"glyph_renderer7":{"id":"1226","type":"GlyphRenderer"},"glyph_renderer8":{"id":"1233","type":"GlyphRenderer"}},"code":"len_labels = 3;glyph_renderers = [[glyph_renderer0,glyph_renderer1,glyph_renderer2],[glyph_renderer3,glyph_renderer4,glyph_renderer5],[glyph_renderer6,glyph_renderer7,glyph_renderer8]];\\n        for (i = 0; i &lt; len_labels; i++) {\\n            if (cb_obj.active.includes(i)) {\\n                // console.log(&#x27;Setting to true: &#x27; + i + &#x27;(&#x27; + glyph_renderers[i].length + &#x27;)&#x27;)\\n                for (j = 0; j &lt; glyph_renderers[i].length; j++) {\\n                    glyph_renderers[i][j].visible = true;\\n                    // console.log(&#x27;Setting to true: &#x27; + i + &#x27; : &#x27; + j)\\n                }\\n            } else {\\n                // console.log(&#x27;Setting to false: &#x27; + i + &#x27;(&#x27; + glyph_renderers[i].length + &#x27;)&#x27;)\\n                for (j = 0; j &lt; glyph_renderers[i].length; j++) {\\n                    glyph_renderers[i][j].visible = false;\\n                    // console.log(&#x27;Setting to false: &#x27; + i + &#x27; : &#x27; + j)\\n                }\\n            }\\n        }\\n        "},"id":"1234","type":"CustomJS"},{"attributes":{"callback":{"id":"1238","type":"CustomJS"},"icon":null,"label":"None"},"id":"1239","type":"Button"},{"attributes":{"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"line_width":{"value":5},"xs":{"field":"times"},"ys":{"field":"losses"}},"id":"1178","type":"MultiLine"},{"attributes":{"fill_alpha":{"value":0.5},"fill_color":{"field":"colors","transform":{"id":"1140","type":"LinearColorMapper"}},"line_color":{"field":"colors"},"size":{"units":"screen","value":20},"x":{"field":"times"},"y":{"field":"losses"}},"id":"1204","type":"CircleX"},{"attributes":{"bounds":"auto","callback":null,"end":9.8,"start":0.19999999999999996},"id":"1141","type":"Range1d"},{"attributes":{"args":{"cm":{"id":"1140","type":"LinearColorMapper"},"source_multiline":{"id":"1138","type":"ColumnDataSource"},"source_scatter":{"id":"1139","type":"ColumnDataSource"}},"code":"\\n            var data_multiline = source_multiline.data;\\n            var data_scatter = source_scatter.data;\\n            var min_perf = 0.14599792659282684;\\n            var max_perf = 1.0928051471710205;\\n            var min_iter = 0;\\n            var max_iter = 2;\\n            if (cb_obj.value == &#x27;performance&#x27;) {\\n                data_multiline[&#x27;colors&#x27;] = data_multiline[&#x27;colors_performance&#x27;];\\n                data_scatter[&#x27;colors&#x27;] = data_scatter[&#x27;colors_performance&#x27;];\\n                cm.low = min_perf;\\n                cm.high = max_perf;\\n            } else {\\n                data_multiline[&#x27;colors&#x27;] = data_multiline[&#x27;colors_iteration&#x27;];\\n                data_scatter[&#x27;colors&#x27;] = data_scatter[&#x27;colors_iteration&#x27;];\\n                cm.low = min_iter;\\n                cm.high = max_iter;\\n            }\\n            source.change.emit();\\n            "},"id":"1240","type":"CustomJS"},{"attributes":{"line_alpha":{"value":0.5},"line_color":{"field":"colors","transform":{"id":"1140","type":"LinearColorMapper"}},"line_width":{"value":5},"xs":{"field":"times"},"ys":{"field":"losses"}},"id":"1177","type":"MultiLine"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":20},"x":{"field":"times"},"y":{"field":"losses"}},"id":"1205","type":"CircleX"},{"attributes":{"callback":{"id":"1240","type":"CustomJS"},"options":["performance","iteration"],"title":"Select colors","value":"performance"},"id":"1241","type":"Select"},{"attributes":{"data_source":{"id":"1139","type":"ColumnDataSource"},"glyph":{"id":"1204","type":"CircleX"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"1205","type":"CircleX"},"selection_glyph":null,"view":{"id":"1202","type":"CDSView"}},"id":"1206","type":"GlyphRenderer"},{"attributes":{"children":[{"id":"1237","type":"Button"},{"id":"1239","type":"Button"}],"width":100},"id":"1242","type":"WidgetBox"},{"attributes":{"bottom_units":"screen","fill_alpha":{"value":0.5},"fill_color":{"value":"lightgrey"},"left_units":"screen","level":"overlay","line_alpha":{"value":1.0},"line_color":{"value":"black"},"line_dash":[4,4],"line_width":{"value":2},"plot":null,"render_mode":"css","right_units":"screen","top_units":"screen"},"id":"1169","type":"BoxAnnotation"},{"attributes":{"column_name":"HB_iteration","group":"1"},"id":"1207","type":"GroupFilter"},{"attributes":{"children":[{"id":"1235","type":"CheckboxButtonGroup"}],"width":500},"id":"1243","type":"WidgetBox"},{"attributes":{"column_name":"config_info","group":"model_based_pick=False"},"id":"1208","type":"GroupFilter"},{"attributes":{"children":[{"id":"1241","type":"Select"}],"width":200},"id":"1244","type":"WidgetBox"},{"attributes":{"filters":[{"id":"1207","type":"GroupFilter"},{"id":"1208","type":"GroupFilter"}],"source":{"id":"1139","type":"ColumnDataSource"}},"id":"1209","type":"CDSView"},{"attributes":{"filters":[{"id":"1174","type":"GroupFilter"}],"source":{"id":"1138","type":"ColumnDataSource"}},"id":"1175","type":"CDSView"},{"attributes":{"children":[{"id":"1242","type":"WidgetBox"},{"id":"1243","type":"WidgetBox"},{"id":"1244","type":"WidgetBox"}]},"id":"1245","type":"Row"},{"attributes":{"below":[{"id":"1150","type":"LinearAxis"}],"left":[{"id":"1155","type":"LogAxis"}],"plot_height":500,"renderers":[{"id":"1150","type":"LinearAxis"},{"id":"1154","type":"Grid"},{"id":"1155","type":"LogAxis"},{"id":"1159","type":"Grid"},{"id":"1169","type":"BoxAnnotation"},{"id":"1179","type":"GlyphRenderer"},{"id":"1186","type":"GlyphRenderer"},{"id":"1193","type":"GlyphRenderer"},{"id":"1199","type":"GlyphRenderer"},{"id":"1206","type":"GlyphRenderer"},{"id":"1213","type":"GlyphRenderer"},{"id":"1219","type":"GlyphRenderer"},{"id":"1226","type":"GlyphRenderer"},{"id":"1233","type":"GlyphRenderer"}],"title":{"id":"1248","type":"Title"},"toolbar":{"id":"1165","type":"Toolbar"},"x_range":{"id":"1141","type":"Range1d"},"x_scale":{"id":"1146","type":"LinearScale"},"y_range":{"id":"1142","type":"Range1d"},"y_scale":{"id":"1148","type":"LogScale"}},"id":"1143","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"1255","type":"Selection"},{"attributes":{"bounds":"auto","callback":null,"end":2.039612367749214,"start":0.13139813393354416},"id":"1142","type":"Range1d"},{"attributes":{"children":[{"id":"1143","subtype":"Figure","type":"Plot"},{"id":"1245","type":"Row"}]},"id":"1246","type":"Column"},{"attributes":{"fill_alpha":{"value":0.5},"fill_color":{"field":"colors","transform":{"id":"1140","type":"LinearColorMapper"}},"line_color":{"field":"colors"},"size":{"units":"screen","value":20},"x":{"field":"times"},"y":{"field":"losses"}},"id":"1211","type":"Circle"},{"attributes":{"column_name":"config_info","group":"model_based_pick=False"},"id":"1188","type":"GroupFilter"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":20},"x":{"field":"times"},"y":{"field":"losses"}},"id":"1212","type":"Circle"},{"attributes":{"data_source":{"id":"1139","type":"ColumnDataSource"},"glyph":{"id":"1211","type":"Circle"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"1212","type":"Circle"},"selection_glyph":null,"view":{"id":"1209","type":"CDSView"}},"id":"1213","type":"GlyphRenderer"},{"attributes":{"column_name":"HB_iteration","group":"0"},"id":"1187","type":"GroupFilter"},{"attributes":{"plot":null,"text":""},"id":"1248","type":"Title"},{"attributes":{"column_name":"HB_iteration","group":"2"},"id":"1214","type":"GroupFilter"},{"attributes":{},"id":"1249","type":"BasicTickFormatter"},{"attributes":{"fill_alpha":{"value":0.5},"fill_color":{"field":"colors","transform":{"id":"1140","type":"LinearColorMapper"}},"line_color":{"field":"colors"},"size":{"units":"screen","value":20},"x":{"field":"times"},"y":{"field":"losses"}},"id":"1184","type":"CircleX"},{"attributes":{"filters":[{"id":"1214","type":"GroupFilter"}],"source":{"id":"1138","type":"ColumnDataSource"}},"id":"1215","type":"CDSView"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":20},"x":{"field":"times"},"y":{"field":"losses"}},"id":"1185","type":"CircleX"},{"attributes":{},"id":"1254","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"1138","type":"ColumnDataSource"},"glyph":{"id":"1177","type":"MultiLine"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"1178","type":"MultiLine"},"selection_glyph":null,"view":{"id":"1175","type":"CDSView"}},"id":"1179","type":"GlyphRenderer"},{"attributes":{"line_alpha":{"value":0.5},"line_color":{"field":"colors","transform":{"id":"1140","type":"LinearColorMapper"}},"line_width":{"value":5},"xs":{"field":"times"},"ys":{"field":"losses"}},"id":"1217","type":"MultiLine"},{"attributes":{"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"line_width":{"value":5},"xs":{"field":"times"},"ys":{"field":"losses"}},"id":"1218","type":"MultiLine"},{"attributes":{"data_source":{"id":"1138","type":"ColumnDataSource"},"glyph":{"id":"1217","type":"MultiLine"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"1218","type":"MultiLine"},"selection_glyph":null,"view":{"id":"1215","type":"CDSView"}},"id":"1219","type":"GlyphRenderer"},{"attributes":{"filters":[{"id":"1180","type":"GroupFilter"},{"id":"1181","type":"GroupFilter"}],"source":{"id":"1139","type":"ColumnDataSource"}},"id":"1182","type":"CDSView"},{"attributes":{"column_name":"HB_iteration","group":"2"},"id":"1220","type":"GroupFilter"},{"attributes":{"column_name":"HB_iteration","group":"0"},"id":"1180","type":"GroupFilter"},{"attributes":{"column_name":"config_info","group":"model_based_pick=True"},"id":"1221","type":"GroupFilter"},{"attributes":{"callback":null,"data":{"HB_iteration":["0","0","0","0","0","0","0","0","0","0","0","0","0","1","1","1","1","2","2","2"],"batch_size":[521,546,659,659,659,102,870,237,237,643,521,221,221,427,300,301,301,350,751,855],"channel_1":[7,16,6,6,6,8,18,14,14,7,17,17,17,15,11,19,19,10,7,16],"channel_2":[4,2,3,3,3,1,3,1,1,3,1,4,4,2,1,2,2,3,4,2],"colors":[0.4955672025680542,0.5255249738693237,0.14599792659282684,0.14599792659282684,0.14599792659282684,0.5820517539978027,0.7636229991912842,0.21017399430274963,0.21017399430274963,1.048065185546875,1.0928051471710205,0.18366342782974243,0.18366342782974243,0.4362681210041046,0.6340452432632446,0.1911911964416504,0.1911911964416504,0.1953982561826706,0.19864468276500702,0.27326250076293945],"colors_iteration":[0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,2],"colors_performance":[0.4955672025680542,0.5255249738693237,0.14599792659282684,0.14599792659282684,0.14599792659282684,0.5820517539978027,0.7636229991912842,0.21017399430274963,0.21017399430274963,1.048065185546875,1.0928051471710205,0.18366342782974243,0.18366342782974243,0.4362681210041046,0.6340452432632446,0.1911911964416504,0.1911911964416504,0.1953982561826706,0.19864468276500702,0.27326250076293945],"config_id":["(0, 0, 0)","(0, 0, 1)","(0, 0, 2)","(0, 0, 2)","(0, 0, 2)","(0, 0, 3)","(0, 0, 4)","(0, 0, 5)","(0, 0, 5)","(0, 0, 6)","(0, 0, 7)","(0, 0, 8)","(0, 0, 8)","(1, 0, 0)","(1, 0, 1)","(1, 0, 2)","(1, 0, 2)","(2, 0, 0)","(2, 0, 1)","(2, 0, 2)"],"config_info":["model_based_pick=False","model_based_pick=False","model_based_pick=False","model_based_pick=False","model_based_pick=False","model_based_pick=False","model_based_pick=False","model_based_pick=False","model_based_pick=False","model_based_pick=False","model_based_pick=False","model_based_pick=False","model_based_pick=False","model_based_pick=False","model_based_pick=False","model_based_pick=False","model_based_pick=False","model_based_pick=False","model_based_pick=False","model_based_pick=False"],"duration":[376.2076714038849,454.88907837867737,1775.6648395061493,1775.6648395061493,1775.6648395061493,152.89501667022705,979.1999363899231,1049.3433637619019,1049.3433637619019,293.56001806259155,493.5110504627228,2499.0606021881104,2499.0606021881104,1349.1332499980927,720.4042983055115,4990.13751578331,4990.13751578331,2761.682498693466,2309.1727414131165,2978.558889389038],"fc_nodes":[518,328,718,718,718,125,"None",557,557,"None",93,268,268,"None","None",101,101,77,114,52],"losses":[0.4955672025680542,0.5255249738693237,0.33489900827407837,0.18270595371723175,0.14599792659282684,0.5820517539978027,0.7636229991912842,0.39340490102767944,0.21017399430274963,1.048065185546875,1.0928051471710205,0.29340508580207825,0.18366342782974243,0.4362681210041046,0.6340452432632446,0.2752169966697693,0.1911911964416504,0.1953982561826706,0.19864468276500702,0.27326250076293945],"n_fc_layer":[2,2,3,3,3,2,1,2,2,1,3,2,2,1,1,2,2,3,3,3],"times":[1.0,1.0,1.0,3.0,9.0,1.0,1.0,1.0,3.0,1.0,1.0,1.0,3.0,3.0,3.0,3.0,9.0,9.0,9.0,9.0]},"selected":{"id":"1257","type":"Selection"},"selection_policy":{"id":"1256","type":"UnionRenderers"}},"id":"1139","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"1139","type":"ColumnDataSource"},"glyph":{"id":"1184","type":"CircleX"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"1185","type":"CircleX"},"selection_glyph":null,"view":{"id":"1182","type":"CDSView"}},"id":"1186","type":"GlyphRenderer"},{"attributes":{"filters":[{"id":"1220","type":"GroupFilter"},{"id":"1221","type":"GroupFilter"}],"source":{"id":"1139","type":"ColumnDataSource"}},"id":"1222","type":"CDSView"},{"attributes":{"callback":null,"renderers":"auto","tooltips":[["config_id","@config_id"],["config_info","@config_info"],["losses","@losses"],["HB_iteration","@HB_iteration"],["duration (sec)","@duration"],["batch_size","@batch_size"],["channel_1","@channel_1"],["channel_2","@channel_2"],["n_fc_layer","@n_fc_layer"],["fc_nodes","@fc_nodes"]]},"id":"1137","type":"HoverTool"},{"attributes":{"high":1.0928051471710205,"low":0.14599792659282684,"palette":["#5e4fa2","#3288bd","#66c2a5","#abdda4","#e6f598","#ffffbf","#fee08b","#fdae61","#f46d43","#d53e4f","#9e0142"]},"id":"1140","type":"LinearColorMapper"},{"attributes":{"ticker":null},"id":"1251","type":"LogTickFormatter"},{"attributes":{"column_name":"config_info","group":"model_based_pick=True"},"id":"1181","type":"GroupFilter"},{"attributes":{"fill_alpha":{"value":0.5},"fill_color":{"field":"colors","transform":{"id":"1140","type":"LinearColorMapper"}},"line_color":{"field":"colors"},"size":{"units":"screen","value":20},"x":{"field":"times"},"y":{"field":"losses"}},"id":"1224","type":"CircleX"},{"attributes":{"callback":null,"data":{"HB_iteration":["0","0","0","0","0","0","0","0","0","1","1","1","2","2","2"],"batch_size":[521,546,659,102,870,237,643,521,221,427,300,301,350,751,855],"channel_1":[7,16,6,8,18,14,7,17,17,15,11,19,10,7,16],"channel_2":[4,2,3,1,3,1,3,1,4,2,1,2,3,4,2],"colors":[0.4955672025680542,0.5255249738693237,0.14599792659282684,0.5820517539978027,0.7636229991912842,0.21017399430274963,1.048065185546875,1.0928051471710205,0.18366342782974243,0.4362681210041046,0.6340452432632446,0.1911911964416504,0.1953982561826706,0.19864468276500702,0.27326250076293945],"colors_iteration":[0,0,0,0,0,0,0,0,0,1,1,1,2,2,2],"colors_performance":[0.4955672025680542,0.5255249738693237,0.14599792659282684,0.5820517539978027,0.7636229991912842,0.21017399430274963,1.048065185546875,1.0928051471710205,0.18366342782974243,0.4362681210041046,0.6340452432632446,0.1911911964416504,0.1953982561826706,0.19864468276500702,0.27326250076293945],"config_id":["(0, 0, 0)","(0, 0, 1)","(0, 0, 2)","(0, 0, 3)","(0, 0, 4)","(0, 0, 5)","(0, 0, 6)","(0, 0, 7)","(0, 0, 8)","(1, 0, 0)","(1, 0, 1)","(1, 0, 2)","(2, 0, 0)","(2, 0, 1)","(2, 0, 2)"],"config_info":["model_based_pick=False","model_based_pick=False","model_based_pick=False","model_based_pick=False","model_based_pick=False","model_based_pick=False","model_based_pick=False","model_based_pick=False","model_based_pick=False","model_based_pick=False","model_based_pick=False","model_based_pick=False","model_based_pick=False","model_based_pick=False","model_based_pick=False"],"duration":[376.2076714038849,454.88907837867737,1775.6648395061493,152.89501667022705,979.1999363899231,1049.3433637619019,293.56001806259155,493.5110504627228,2499.0606021881104,1349.1332499980927,720.4042983055115,4990.13751578331,2761.682498693466,2309.1727414131165,2978.558889389038],"fc_nodes":[518,328,718,125,"None",557,"None",93,268,"None","None",101,77,114,52],"losses":[[0.4955672025680542],[0.5255249738693237],[0.33489900827407837,0.18270595371723175,0.14599792659282684],[0.5820517539978027],[0.7636229991912842],[0.39340490102767944,0.21017399430274963],[1.048065185546875],[1.0928051471710205],[0.29340508580207825,0.18366342782974243],[0.4362681210041046],[0.6340452432632446],[0.2752169966697693,0.1911911964416504],[0.1953982561826706],[0.19864468276500702],[0.27326250076293945]],"n_fc_layer":[2,2,3,2,1,2,1,3,2,1,1,2,3,3,3],"times":[[1.0],[1.0],[1.0,3.0,9.0],[1.0],[1.0],[1.0,3.0],[1.0],[1.0],[1.0,3.0],[3.0],[3.0],[3.0,9.0],[9.0],[9.0],[9.0]]},"selected":{"id":"1255","type":"Selection"},"selection_policy":{"id":"1254","type":"UnionRenderers"}},"id":"1138","type":"ColumnDataSource"}],"root_ids":["1246"]},"title":"Bokeh Application","version":"1.0.1"}}';
          var render_items = [{"docid":"13b633bc-7c0e-4990-9d78-5227b097850d","roots":{"1246":"6b5a94b7-a113-4ac2-a4a3-ab374828cdda"}}];
          root.Bokeh.embed.embed_items(docs_json, render_items);
        
          }
          if (root.Bokeh !== undefined) {
            embed_document(root);
          } else {
            var attempts = 0;
            var timer = setInterval(function(root) {
              if (root.Bokeh !== undefined) {
                embed_document(root);
                clearInterval(timer);
              }
              attempts++;
              if (attempts > 100) {
                console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                clearInterval(timer);
              }
            }, 10, root)
          }
        })(window);
      });
    };
    if (document.readyState != "loading") fn();
    else document.addEventListener("DOMContentLoaded", fn);
  })();