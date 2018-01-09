
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Javascript template for HTMLWriter
2: JS_INCLUDE = '''
3: <link rel="stylesheet"
4: href="https://maxcdn.bootstrapcdn.com/font-awesome/4.4.0/
5: css/font-awesome.min.css">
6: <script language="javascript">
7:   /* Define the Animation class */
8:   function Animation(frames, img_id, slider_id, interval, loop_select_id){
9:     this.img_id = img_id;
10:     this.slider_id = slider_id;
11:     this.loop_select_id = loop_select_id;
12:     this.interval = interval;
13:     this.current_frame = 0;
14:     this.direction = 0;
15:     this.timer = null;
16:     this.frames = new Array(frames.length);
17: 
18:     for (var i=0; i<frames.length; i++)
19:     {
20:      this.frames[i] = new Image();
21:      this.frames[i].src = frames[i];
22:     }
23:     document.getElementById(this.slider_id).max = this.frames.length - 1;
24:     this.set_frame(this.current_frame);
25:   }
26: 
27:   Animation.prototype.get_loop_state = function(){
28:     var button_group = document[this.loop_select_id].state;
29:     for (var i = 0; i < button_group.length; i++) {
30:         var button = button_group[i];
31:         if (button.checked) {
32:             return button.value;
33:         }
34:     }
35:     return undefined;
36:   }
37: 
38:   Animation.prototype.set_frame = function(frame){
39:     this.current_frame = frame;
40:     document.getElementById(this.img_id).src =
41:             this.frames[this.current_frame].src;
42:     document.getElementById(this.slider_id).value = this.current_frame;
43:   }
44: 
45:   Animation.prototype.next_frame = function()
46:   {
47:     this.set_frame(Math.min(this.frames.length - 1, this.current_frame + 1));
48:   }
49: 
50:   Animation.prototype.previous_frame = function()
51:   {
52:     this.set_frame(Math.max(0, this.current_frame - 1));
53:   }
54: 
55:   Animation.prototype.first_frame = function()
56:   {
57:     this.set_frame(0);
58:   }
59: 
60:   Animation.prototype.last_frame = function()
61:   {
62:     this.set_frame(this.frames.length - 1);
63:   }
64: 
65:   Animation.prototype.slower = function()
66:   {
67:     this.interval /= 0.7;
68:     if(this.direction > 0){this.play_animation();}
69:     else if(this.direction < 0){this.reverse_animation();}
70:   }
71: 
72:   Animation.prototype.faster = function()
73:   {
74:     this.interval *= 0.7;
75:     if(this.direction > 0){this.play_animation();}
76:     else if(this.direction < 0){this.reverse_animation();}
77:   }
78: 
79:   Animation.prototype.anim_step_forward = function()
80:   {
81:     this.current_frame += 1;
82:     if(this.current_frame < this.frames.length){
83:       this.set_frame(this.current_frame);
84:     }else{
85:       var loop_state = this.get_loop_state();
86:       if(loop_state == "loop"){
87:         this.first_frame();
88:       }else if(loop_state == "reflect"){
89:         this.last_frame();
90:         this.reverse_animation();
91:       }else{
92:         this.pause_animation();
93:         this.last_frame();
94:       }
95:     }
96:   }
97: 
98:   Animation.prototype.anim_step_reverse = function()
99:   {
100:     this.current_frame -= 1;
101:     if(this.current_frame >= 0){
102:       this.set_frame(this.current_frame);
103:     }else{
104:       var loop_state = this.get_loop_state();
105:       if(loop_state == "loop"){
106:         this.last_frame();
107:       }else if(loop_state == "reflect"){
108:         this.first_frame();
109:         this.play_animation();
110:       }else{
111:         this.pause_animation();
112:         this.first_frame();
113:       }
114:     }
115:   }
116: 
117:   Animation.prototype.pause_animation = function()
118:   {
119:     this.direction = 0;
120:     if (this.timer){
121:       clearInterval(this.timer);
122:       this.timer = null;
123:     }
124:   }
125: 
126:   Animation.prototype.play_animation = function()
127:   {
128:     this.pause_animation();
129:     this.direction = 1;
130:     var t = this;
131:     if (!this.timer) this.timer = setInterval(function() {
132:         t.anim_step_forward();
133:     }, this.interval);
134:   }
135: 
136:   Animation.prototype.reverse_animation = function()
137:   {
138:     this.pause_animation();
139:     this.direction = -1;
140:     var t = this;
141:     if (!this.timer) this.timer = setInterval(function() {
142:         t.anim_step_reverse();
143:     }, this.interval);
144:   }
145: </script>
146: '''
147: 
148: 
149: # HTML template for HTMLWriter
150: DISPLAY_TEMPLATE = '''
151: <div class="animation" align="center">
152:     <img id="_anim_img{id}">
153:     <br>
154:     <input id="_anim_slider{id}" type="range" style="width:350px"
155:            name="points" min="0" max="1" step="1" value="0"
156:            onchange="anim{id}.set_frame(parseInt(this.value));"></input>
157:     <br>
158:     <button onclick="anim{id}.slower()"><i class="fa fa-minus"></i></button>
159:     <button onclick="anim{id}.first_frame()"><i class="fa fa-fast-backward">
160:         </i></button>
161:     <button onclick="anim{id}.previous_frame()">
162:         <i class="fa fa-step-backward"></i></button>
163:     <button onclick="anim{id}.reverse_animation()">
164:         <i class="fa fa-play fa-flip-horizontal"></i></button>
165:     <button onclick="anim{id}.pause_animation()"><i class="fa fa-pause">
166:         </i></button>
167:     <button onclick="anim{id}.play_animation()"><i class="fa fa-play"></i>
168:         </button>
169:     <button onclick="anim{id}.next_frame()"><i class="fa fa-step-forward">
170:         </i></button>
171:     <button onclick="anim{id}.last_frame()"><i class="fa fa-fast-forward">
172:         </i></button>
173:     <button onclick="anim{id}.faster()"><i class="fa fa-plus"></i></button>
174:   <form action="#n" name="_anim_loop_select{id}" class="anim_control">
175:     <input type="radio" name="state"
176:            value="once" {once_checked}> Once </input>
177:     <input type="radio" name="state"
178:            value="loop" {loop_checked}> Loop </input>
179:     <input type="radio" name="state"
180:            value="reflect" {reflect_checked}> Reflect </input>
181:   </form>
182: </div>
183: 
184: 
185: <script language="javascript">
186:   /* Instantiate the Animation class. */
187:   /* The IDs given should match those used in the template above. */
188:   (function() {{
189:     var img_id = "_anim_img{id}";
190:     var slider_id = "_anim_slider{id}";
191:     var loop_select_id = "_anim_loop_select{id}";
192:     var frames = new Array({Nframes});
193:     {fill_frames}
194: 
195:     /* set a timeout to make sure all the above elements are created before
196:        the object is initialized. */
197:     setTimeout(function() {{
198:         anim{id} = new Animation(frames, img_id, slider_id, {interval},
199:                                  loop_select_id);
200:     }}, 0);
201:   }})()
202: </script>
203: '''
204: 
205: INCLUDED_FRAMES = '''
206:   for (var i=0; i<{Nframes}; i++){{
207:     frames[i] = "{frame_dir}/frame" + ("0000000" + i).slice(-7) +
208:                 ".{frame_format}";
209:   }}
210: '''
211: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_169929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, (-1)), 'str', '\n<link rel="stylesheet"\nhref="https://maxcdn.bootstrapcdn.com/font-awesome/4.4.0/\ncss/font-awesome.min.css">\n<script language="javascript">\n  /* Define the Animation class */\n  function Animation(frames, img_id, slider_id, interval, loop_select_id){\n    this.img_id = img_id;\n    this.slider_id = slider_id;\n    this.loop_select_id = loop_select_id;\n    this.interval = interval;\n    this.current_frame = 0;\n    this.direction = 0;\n    this.timer = null;\n    this.frames = new Array(frames.length);\n\n    for (var i=0; i<frames.length; i++)\n    {\n     this.frames[i] = new Image();\n     this.frames[i].src = frames[i];\n    }\n    document.getElementById(this.slider_id).max = this.frames.length - 1;\n    this.set_frame(this.current_frame);\n  }\n\n  Animation.prototype.get_loop_state = function(){\n    var button_group = document[this.loop_select_id].state;\n    for (var i = 0; i < button_group.length; i++) {\n        var button = button_group[i];\n        if (button.checked) {\n            return button.value;\n        }\n    }\n    return undefined;\n  }\n\n  Animation.prototype.set_frame = function(frame){\n    this.current_frame = frame;\n    document.getElementById(this.img_id).src =\n            this.frames[this.current_frame].src;\n    document.getElementById(this.slider_id).value = this.current_frame;\n  }\n\n  Animation.prototype.next_frame = function()\n  {\n    this.set_frame(Math.min(this.frames.length - 1, this.current_frame + 1));\n  }\n\n  Animation.prototype.previous_frame = function()\n  {\n    this.set_frame(Math.max(0, this.current_frame - 1));\n  }\n\n  Animation.prototype.first_frame = function()\n  {\n    this.set_frame(0);\n  }\n\n  Animation.prototype.last_frame = function()\n  {\n    this.set_frame(this.frames.length - 1);\n  }\n\n  Animation.prototype.slower = function()\n  {\n    this.interval /= 0.7;\n    if(this.direction > 0){this.play_animation();}\n    else if(this.direction < 0){this.reverse_animation();}\n  }\n\n  Animation.prototype.faster = function()\n  {\n    this.interval *= 0.7;\n    if(this.direction > 0){this.play_animation();}\n    else if(this.direction < 0){this.reverse_animation();}\n  }\n\n  Animation.prototype.anim_step_forward = function()\n  {\n    this.current_frame += 1;\n    if(this.current_frame < this.frames.length){\n      this.set_frame(this.current_frame);\n    }else{\n      var loop_state = this.get_loop_state();\n      if(loop_state == "loop"){\n        this.first_frame();\n      }else if(loop_state == "reflect"){\n        this.last_frame();\n        this.reverse_animation();\n      }else{\n        this.pause_animation();\n        this.last_frame();\n      }\n    }\n  }\n\n  Animation.prototype.anim_step_reverse = function()\n  {\n    this.current_frame -= 1;\n    if(this.current_frame >= 0){\n      this.set_frame(this.current_frame);\n    }else{\n      var loop_state = this.get_loop_state();\n      if(loop_state == "loop"){\n        this.last_frame();\n      }else if(loop_state == "reflect"){\n        this.first_frame();\n        this.play_animation();\n      }else{\n        this.pause_animation();\n        this.first_frame();\n      }\n    }\n  }\n\n  Animation.prototype.pause_animation = function()\n  {\n    this.direction = 0;\n    if (this.timer){\n      clearInterval(this.timer);\n      this.timer = null;\n    }\n  }\n\n  Animation.prototype.play_animation = function()\n  {\n    this.pause_animation();\n    this.direction = 1;\n    var t = this;\n    if (!this.timer) this.timer = setInterval(function() {\n        t.anim_step_forward();\n    }, this.interval);\n  }\n\n  Animation.prototype.reverse_animation = function()\n  {\n    this.pause_animation();\n    this.direction = -1;\n    var t = this;\n    if (!this.timer) this.timer = setInterval(function() {\n        t.anim_step_reverse();\n    }, this.interval);\n  }\n</script>\n')
# Assigning a type to the variable 'JS_INCLUDE' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'JS_INCLUDE', str_169929)

# Assigning a Str to a Name (line 150):
str_169930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, (-1)), 'str', '\n<div class="animation" align="center">\n    <img id="_anim_img{id}">\n    <br>\n    <input id="_anim_slider{id}" type="range" style="width:350px"\n           name="points" min="0" max="1" step="1" value="0"\n           onchange="anim{id}.set_frame(parseInt(this.value));"></input>\n    <br>\n    <button onclick="anim{id}.slower()"><i class="fa fa-minus"></i></button>\n    <button onclick="anim{id}.first_frame()"><i class="fa fa-fast-backward">\n        </i></button>\n    <button onclick="anim{id}.previous_frame()">\n        <i class="fa fa-step-backward"></i></button>\n    <button onclick="anim{id}.reverse_animation()">\n        <i class="fa fa-play fa-flip-horizontal"></i></button>\n    <button onclick="anim{id}.pause_animation()"><i class="fa fa-pause">\n        </i></button>\n    <button onclick="anim{id}.play_animation()"><i class="fa fa-play"></i>\n        </button>\n    <button onclick="anim{id}.next_frame()"><i class="fa fa-step-forward">\n        </i></button>\n    <button onclick="anim{id}.last_frame()"><i class="fa fa-fast-forward">\n        </i></button>\n    <button onclick="anim{id}.faster()"><i class="fa fa-plus"></i></button>\n  <form action="#n" name="_anim_loop_select{id}" class="anim_control">\n    <input type="radio" name="state"\n           value="once" {once_checked}> Once </input>\n    <input type="radio" name="state"\n           value="loop" {loop_checked}> Loop </input>\n    <input type="radio" name="state"\n           value="reflect" {reflect_checked}> Reflect </input>\n  </form>\n</div>\n\n\n<script language="javascript">\n  /* Instantiate the Animation class. */\n  /* The IDs given should match those used in the template above. */\n  (function() {{\n    var img_id = "_anim_img{id}";\n    var slider_id = "_anim_slider{id}";\n    var loop_select_id = "_anim_loop_select{id}";\n    var frames = new Array({Nframes});\n    {fill_frames}\n\n    /* set a timeout to make sure all the above elements are created before\n       the object is initialized. */\n    setTimeout(function() {{\n        anim{id} = new Animation(frames, img_id, slider_id, {interval},\n                                 loop_select_id);\n    }}, 0);\n  }})()\n</script>\n')
# Assigning a type to the variable 'DISPLAY_TEMPLATE' (line 150)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 0), 'DISPLAY_TEMPLATE', str_169930)

# Assigning a Str to a Name (line 205):
str_169931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, (-1)), 'str', '\n  for (var i=0; i<{Nframes}; i++){{\n    frames[i] = "{frame_dir}/frame" + ("0000000" + i).slice(-7) +\n                ".{frame_format}";\n  }}\n')
# Assigning a type to the variable 'INCLUDED_FRAMES' (line 205)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 0), 'INCLUDED_FRAMES', str_169931)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
