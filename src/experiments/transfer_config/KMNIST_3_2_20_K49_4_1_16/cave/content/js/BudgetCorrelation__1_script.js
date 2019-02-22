
  (function() {
    var fn = function() {
      Bokeh.safely(function() {
        (function(root) {
          function embed_document(root) {
            
          var docs_json = '{"aad434bc-7e6b-4eff-8bea-fb019baac875":{"roots":{"references":[{"attributes":{},"id":"1016","type":"LinearScale"},{"attributes":{"axis_label":"budget","formatter":{"id":"1055","type":"BasicTickFormatter"},"plot":{"id":"1011","subtype":"Figure","type":"Plot"},"ticker":{"id":"1019","type":"BasicTicker"}},"id":"1018","type":"LinearAxis"},{"attributes":{"columns":[{"id":"1002","type":"TableColumn"},{"id":"1003","type":"TableColumn"},{"id":"1004","type":"TableColumn"},{"id":"1005","type":"TableColumn"}],"height":110,"index_position":null,"sortable":false,"source":{"id":"1001","type":"ColumnDataSource"},"view":{"id":"1007","type":"CDSView"}},"id":"1006","type":"DataTable"},{"attributes":{"editor":{"id":"1060","type":"StringEditor"},"field":"Budget","formatter":{"id":"1061","type":"StringFormatter"},"sortable":false,"title":"Budget","width":20},"id":"1002","type":"TableColumn"},{"attributes":{},"id":"1019","type":"BasicTicker"},{"attributes":{"plot":{"id":"1011","subtype":"Figure","type":"Plot"},"ticker":{"id":"1019","type":"BasicTicker"}},"id":"1022","type":"Grid"},{"attributes":{"axis_label":"budget","formatter":{"id":"1057","type":"BasicTickFormatter"},"plot":{"id":"1011","subtype":"Figure","type":"Plot"},"ticker":{"id":"1024","type":"BasicTicker"}},"id":"1023","type":"LinearAxis"},{"attributes":{"data_source":{"id":"1008","type":"ColumnDataSource"},"glyph":{"id":"1044","type":"Circle"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"1045","type":"Circle"},"selection_glyph":null,"view":{"id":"1047","type":"CDSView"}},"id":"1046","type":"GlyphRenderer"},{"attributes":{},"id":"1055","type":"BasicTickFormatter"},{"attributes":{},"id":"1024","type":"BasicTicker"},{"attributes":{"children":[{"id":"1006","type":"DataTable"}],"height":110},"id":"1050","type":"WidgetBox"},{"attributes":{},"id":"1063","type":"StringFormatter"},{"attributes":{"dimension":1,"plot":{"id":"1011","subtype":"Figure","type":"Plot"},"ticker":{"id":"1024","type":"BasicTicker"}},"id":"1027","type":"Grid"},{"attributes":{},"id":"1057","type":"BasicTickFormatter"},{"attributes":{},"id":"1064","type":"StringEditor"},{"attributes":{"js_property_callbacks":{"change:indices":[{"id":"1048","type":"CustomJS"}]}},"id":"1049","type":"Selection"},{"attributes":{},"id":"1061","type":"StringFormatter"},{"attributes":{},"id":"1062","type":"StringEditor"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"1028","type":"PanTool"},{"id":"1029","type":"WheelZoomTool"},{"id":"1030","type":"BoxZoomTool"},{"id":"1031","type":"SaveTool"},{"id":"1032","type":"ResetTool"},{"id":"1033","type":"HelpTool"}]},"id":"1034","type":"Toolbar"},{"attributes":{"children":[{"id":"1050","type":"WidgetBox"},{"id":"1011","subtype":"Figure","type":"Plot"}]},"id":"1051","type":"Column"},{"attributes":{},"id":"1065","type":"StringFormatter"},{"attributes":{},"id":"1066","type":"StringEditor"},{"attributes":{"plot":null,"text":""},"id":"1053","type":"Title"},{"attributes":{"default_sort":"descending","editor":{"id":"1066","type":"StringEditor"},"field":"budget_16.0","formatter":{"id":"1067","type":"StringFormatter"},"title":"budget_16.0","width":10},"id":"1005","type":"TableColumn"},{"attributes":{},"id":"1028","type":"PanTool"},{"attributes":{"default_sort":"descending","editor":{"id":"1064","type":"StringEditor"},"field":"budget_4.0","formatter":{"id":"1065","type":"StringFormatter"},"title":"budget_4.0","width":10},"id":"1004","type":"TableColumn"},{"attributes":{},"id":"1060","type":"StringEditor"},{"attributes":{},"id":"1058","type":"UnionRenderers"},{"attributes":{},"id":"1067","type":"StringFormatter"},{"attributes":{"source":{"id":"1001","type":"ColumnDataSource"}},"id":"1007","type":"CDSView"},{"attributes":{},"id":"1029","type":"WheelZoomTool"},{"attributes":{},"id":"1068","type":"UnionRenderers"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":5},"x":{"field":"x"},"y":{"field":"y"}},"id":"1045","type":"Circle"},{"attributes":{"overlay":{"id":"1036","type":"BoxAnnotation"}},"id":"1030","type":"BoxZoomTool"},{"attributes":{},"id":"1069","type":"Selection"},{"attributes":{"bounds":[0.028963740170001986,1.1521075263619422],"callback":null,"end":1.1521075263619422,"start":0.028963740170001986},"id":"1009","type":"Range1d"},{"attributes":{"fill_alpha":{"value":0.5},"fill_color":{"value":"navy"},"line_alpha":{"value":0.5},"line_color":{"value":"navy"},"size":{"units":"screen","value":5},"x":{"field":"x"},"y":{"field":"y"}},"id":"1044","type":"Circle"},{"attributes":{"callback":null,"data":{"budget_1.0":[0.39241650700569153,0.2951762080192566,0.5533370971679688,0.4577952027320862,0.4411420226097107,0.647633969783783,0.33505719900131226,null,0.38558632135391235,1.0585122108459473,0.558938205242157,0.4244753420352936,0.3059351146221161,null,0.8496960997581482,0.2705405056476593,0.3168765604496002,0.32803958654403687,0.33757829666137695,null,0.2928747832775116,0.29196467995643616,0.5479579567909241,0.9187172651290894,0.9849998950958252,0.44716712832450867,0.3116127848625183,null,0.27532684803009033,0.32323771715164185,null,null,0.5265423655509949,0.28421491384506226,null,0.8286634087562561,0.5956897735595703,0.2898262143135071,0.9020982980728149,0.5503836274147034,0.30634209513664246,0.9812561273574829,0.8199624419212341,0.511613667011261,null,0.41422250866889954,null,0.3334220051765442,0.3803691267967224,0.4288066625595093,0.3324647843837738,0.40570276975631714,null,null,0.3869388997554779,0.28428345918655396,0.48453813791275024,null,null,0.4309160113334656,null],"budget_16.0":[null,null,null,null,null,null,null,0.17250430583953857,null,null,null,null,0.12402111291885376,0.12348770350217819,null,0.12255905568599701,null,null,null,null,null,0.13129621744155884,null,null,null,null,null,null,null,null,null,0.24238286912441254,null,null,0.1683443784713745,null,null,null,null,null,null,null,null,null,0.12809009850025177,null,0.24554340541362762,null,null,null,null,null,null,null,null,null,null,0.1352853924036026,0.21142829954624176,null,null],"budget_4.0":[null,null,null,null,0.2686487138271332,null,0.20259009301662445,null,null,null,null,null,0.1711500883102417,0.16975876688957214,null,0.15713617205619812,0.1740475744009018,null,null,0.20080110430717468,null,0.171990767121315,null,null,null,null,null,0.2020723819732666,0.16029703617095947,null,0.3704269826412201,null,null,0.1695164144039154,null,null,null,null,null,null,null,null,null,null,null,null,null,0.1790102869272232,0.22674676775932312,null,null,null,0.34085115790367126,0.18132659792900085,0.21974265575408936,0.175335094332695,null,0.17942574620246887,null,null,0.2716793119907379],"x":[],"y":[]},"selected":{"id":"1069","type":"Selection"},"selection_policy":{"id":"1068","type":"UnionRenderers"}},"id":"1008","type":"ColumnDataSource"},{"attributes":{},"id":"1031","type":"SaveTool"},{"attributes":{"default_sort":"descending","editor":{"id":"1062","type":"StringEditor"},"field":"budget_1.0","formatter":{"id":"1063","type":"StringFormatter"},"title":"budget_1.0","width":10},"id":"1003","type":"TableColumn"},{"attributes":{"bottom_units":"screen","fill_alpha":{"value":0.5},"fill_color":{"value":"lightgrey"},"left_units":"screen","level":"overlay","line_alpha":{"value":1.0},"line_color":{"value":"black"},"line_dash":[4,4],"line_width":{"value":2},"plot":null,"render_mode":"css","right_units":"screen","top_units":"screen"},"id":"1036","type":"BoxAnnotation"},{"attributes":{},"id":"1032","type":"ResetTool"},{"attributes":{"bounds":[0.028963740170001986,1.1521075263619422],"callback":null,"end":1.1521075263619422,"start":0.028963740170001986},"id":"1010","type":"Range1d"},{"attributes":{},"id":"1033","type":"HelpTool"},{"attributes":{"below":[{"id":"1018","type":"LinearAxis"}],"left":[{"id":"1023","type":"LinearAxis"}],"match_aspect":true,"plot_height":400,"plot_width":400,"renderers":[{"id":"1018","type":"LinearAxis"},{"id":"1022","type":"Grid"},{"id":"1023","type":"LinearAxis"},{"id":"1027","type":"Grid"},{"id":"1036","type":"BoxAnnotation"},{"id":"1046","type":"GlyphRenderer"}],"title":{"id":"1053","type":"Title"},"toolbar":{"id":"1034","type":"Toolbar"},"x_range":{"id":"1010","type":"Range1d"},"x_scale":{"id":"1014","type":"LinearScale"},"y_range":{"id":"1009","type":"Range1d"},"y_scale":{"id":"1016","type":"LinearScale"}},"id":"1011","subtype":"Figure","type":"Plot"},{"attributes":{"source":{"id":"1008","type":"ColumnDataSource"}},"id":"1047","type":"CDSView"},{"attributes":{},"id":"1014","type":"LinearScale"},{"attributes":{"args":{"scatter_source":{"id":"1008","type":"ColumnDataSource"},"table_source":{"id":"1001","type":"ColumnDataSource"},"xaxis":{"id":"1018","type":"LinearAxis"},"xr":{"id":"1010","type":"Range1d"},"yaxis":{"id":"1023","type":"LinearAxis"},"yr":{"id":"1009","type":"Range1d"}},"code":"var budgets = [&#x27;budget_1.0&#x27;, &#x27;budget_4.0&#x27;, &#x27;budget_16.0&#x27;];console.log(budgets);\\n        try {\\n            // This first part only extracts selected row and column!\\n            var grid = document.getElementsByClassName(&#x27;grid-canvas&#x27;)[0].children;\\n            var row = &#x27;&#x27;;\\n            var col = &#x27;&#x27;;\\n            for (var i=0,max=grid.length;i&lt;max;i++){\\n                if (grid[i].outerHTML.includes(&#x27;active&#x27;)){\\n                    row=i;\\n                    for (var j=0,jmax=grid[i].children.length;j&lt;jmax;j++){\\n                        if(grid[i].children[j].outerHTML.includes(&#x27;active&#x27;)){col=j}\\n                    }\\n                }\\n            }\\n            col = col - 1;\\n            console.log(&#x27;row&#x27;, row, budgets[row]);\\n            console.log(&#x27;col&#x27;, col, budgets[col]);\\n            table_source.selected.indices = [];  // Reset, so gets triggered again when clicked again\\n\\n            // This is the actual updating of the plot\\n            if (row =&gt;  0 &amp;&amp; col &gt; 0) {\\n              // Copy relevant arrays\\n              var new_x = scatter_source.data[budgets[row]].slice();\\n              var new_y = scatter_source.data[budgets[col]].slice();\\n              // Remove all pairs where one value is null\\n              while ((next_null = new_x.indexOf(null)) &gt; -1) {\\n                new_x.splice(next_null, 1);\\n                new_y.splice(next_null, 1);\\n              }\\n              while ((next_null = new_y.indexOf(null)) &gt; -1) {\\n                new_x.splice(next_null, 1);\\n                new_y.splice(next_null, 1);\\n              }\\n              // Assign new data to the plotted columns\\n              scatter_source.data[&#x27;x&#x27;] = new_x;\\n              scatter_source.data[&#x27;y&#x27;] = new_y;\\n              scatter_source.change.emit();\\n              // Update axis-labels\\n              xaxis.attributes.axis_label = budgets[row];\\n              yaxis.attributes.axis_label = budgets[col];\\n              // Update ranges\\n              var min = Math.min(...[Math.min(...new_x), Math.min(...new_y)])\\n                  max = Math.max(...[Math.max(...new_x), Math.max(...new_y)]);\\n              var padding = (max - min) / 10;\\n              console.log(min, max, padding);\\n              xr.start = min - padding;\\n              yr.start = min - padding;\\n              xr.end = max + padding;\\n              yr.end = max + padding;\\n            }\\n        } catch(err) {\\n            console.log(err.message);\\n        }\\n        "},"id":"1048","type":"CustomJS"},{"attributes":{"callback":null,"data":{"Budget":["budget_1.0","budget_4.0","budget_16.0"],"budget_1.0":["1.00 (47 samples)","",""],"budget_16.0":["0.50 (3 samples)","1.00 (5 samples)","1.00 (11 samples)"],"budget_4.0":["0.94 (12 samples)","1.00 (20 samples)",""]},"selected":{"id":"1049","type":"Selection"},"selection_policy":{"id":"1058","type":"UnionRenderers"}},"id":"1001","type":"ColumnDataSource"}],"root_ids":["1051"]},"title":"Bokeh Application","version":"1.0.1"}}';
          var render_items = [{"docid":"aad434bc-7e6b-4eff-8bea-fb019baac875","roots":{"1051":"a6a87e9c-d0a5-4fee-8857-84a3c8fff900"}}];
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
