
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: from collections import OrderedDict
4: import six
5: 
6: 
7: BASE_COLORS = {
8:     'b': (0, 0, 1),
9:     'g': (0, 0.5, 0),
10:     'r': (1, 0, 0),
11:     'c': (0, 0.75, 0.75),
12:     'm': (0.75, 0, 0.75),
13:     'y': (0.75, 0.75, 0),
14:     'k': (0, 0, 0),
15:     'w': (1, 1, 1)}
16: 
17: 
18: # These colors are from Tableau
19: TABLEAU_COLORS = (
20:     ('blue', '#1f77b4'),
21:     ('orange', '#ff7f0e'),
22:     ('green', '#2ca02c'),
23:     ('red', '#d62728'),
24:     ('purple', '#9467bd'),
25:     ('brown', '#8c564b'),
26:     ('pink', '#e377c2'),
27:     ('gray', '#7f7f7f'),
28:     ('olive', '#bcbd22'),
29:     ('cyan', '#17becf'),
30: )
31: 
32: # Normalize name to "tab:<name>" to avoid name collisions.
33: TABLEAU_COLORS = OrderedDict(
34:     ('tab:' + name, value) for name, value in TABLEAU_COLORS)
35: 
36: # This mapping of color names -> hex values is taken from
37: # a survey run by Randel Monroe see:
38: # http://blog.xkcd.com/2010/05/03/color-survey-results/
39: # for more details.  The results are hosted at
40: # https://xkcd.com/color/rgb.txt
41: #
42: # License: http://creativecommons.org/publicdomain/zero/1.0/
43: XKCD_COLORS = {
44:     'cloudy blue': '#acc2d9',
45:     'dark pastel green': '#56ae57',
46:     'dust': '#b2996e',
47:     'electric lime': '#a8ff04',
48:     'fresh green': '#69d84f',
49:     'light eggplant': '#894585',
50:     'nasty green': '#70b23f',
51:     'really light blue': '#d4ffff',
52:     'tea': '#65ab7c',
53:     'warm purple': '#952e8f',
54:     'yellowish tan': '#fcfc81',
55:     'cement': '#a5a391',
56:     'dark grass green': '#388004',
57:     'dusty teal': '#4c9085',
58:     'grey teal': '#5e9b8a',
59:     'macaroni and cheese': '#efb435',
60:     'pinkish tan': '#d99b82',
61:     'spruce': '#0a5f38',
62:     'strong blue': '#0c06f7',
63:     'toxic green': '#61de2a',
64:     'windows blue': '#3778bf',
65:     'blue blue': '#2242c7',
66:     'blue with a hint of purple': '#533cc6',
67:     'booger': '#9bb53c',
68:     'bright sea green': '#05ffa6',
69:     'dark green blue': '#1f6357',
70:     'deep turquoise': '#017374',
71:     'green teal': '#0cb577',
72:     'strong pink': '#ff0789',
73:     'bland': '#afa88b',
74:     'deep aqua': '#08787f',
75:     'lavender pink': '#dd85d7',
76:     'light moss green': '#a6c875',
77:     'light seafoam green': '#a7ffb5',
78:     'olive yellow': '#c2b709',
79:     'pig pink': '#e78ea5',
80:     'deep lilac': '#966ebd',
81:     'desert': '#ccad60',
82:     'dusty lavender': '#ac86a8',
83:     'purpley grey': '#947e94',
84:     'purply': '#983fb2',
85:     'candy pink': '#ff63e9',
86:     'light pastel green': '#b2fba5',
87:     'boring green': '#63b365',
88:     'kiwi green': '#8ee53f',
89:     'light grey green': '#b7e1a1',
90:     'orange pink': '#ff6f52',
91:     'tea green': '#bdf8a3',
92:     'very light brown': '#d3b683',
93:     'egg shell': '#fffcc4',
94:     'eggplant purple': '#430541',
95:     'powder pink': '#ffb2d0',
96:     'reddish grey': '#997570',
97:     'baby shit brown': '#ad900d',
98:     'liliac': '#c48efd',
99:     'stormy blue': '#507b9c',
100:     'ugly brown': '#7d7103',
101:     'custard': '#fffd78',
102:     'darkish pink': '#da467d',
103:     'deep brown': '#410200',
104:     'greenish beige': '#c9d179',
105:     'manilla': '#fffa86',
106:     'off blue': '#5684ae',
107:     'battleship grey': '#6b7c85',
108:     'browny green': '#6f6c0a',
109:     'bruise': '#7e4071',
110:     'kelley green': '#009337',
111:     'sickly yellow': '#d0e429',
112:     'sunny yellow': '#fff917',
113:     'azul': '#1d5dec',
114:     'darkgreen': '#054907',
115:     'green/yellow': '#b5ce08',
116:     'lichen': '#8fb67b',
117:     'light light green': '#c8ffb0',
118:     'pale gold': '#fdde6c',
119:     'sun yellow': '#ffdf22',
120:     'tan green': '#a9be70',
121:     'burple': '#6832e3',
122:     'butterscotch': '#fdb147',
123:     'toupe': '#c7ac7d',
124:     'dark cream': '#fff39a',
125:     'indian red': '#850e04',
126:     'light lavendar': '#efc0fe',
127:     'poison green': '#40fd14',
128:     'baby puke green': '#b6c406',
129:     'bright yellow green': '#9dff00',
130:     'charcoal grey': '#3c4142',
131:     'squash': '#f2ab15',
132:     'cinnamon': '#ac4f06',
133:     'light pea green': '#c4fe82',
134:     'radioactive green': '#2cfa1f',
135:     'raw sienna': '#9a6200',
136:     'baby purple': '#ca9bf7',
137:     'cocoa': '#875f42',
138:     'light royal blue': '#3a2efe',
139:     'orangeish': '#fd8d49',
140:     'rust brown': '#8b3103',
141:     'sand brown': '#cba560',
142:     'swamp': '#698339',
143:     'tealish green': '#0cdc73',
144:     'burnt siena': '#b75203',
145:     'camo': '#7f8f4e',
146:     'dusk blue': '#26538d',
147:     'fern': '#63a950',
148:     'old rose': '#c87f89',
149:     'pale light green': '#b1fc99',
150:     'peachy pink': '#ff9a8a',
151:     'rosy pink': '#f6688e',
152:     'light bluish green': '#76fda8',
153:     'light bright green': '#53fe5c',
154:     'light neon green': '#4efd54',
155:     'light seafoam': '#a0febf',
156:     'tiffany blue': '#7bf2da',
157:     'washed out green': '#bcf5a6',
158:     'browny orange': '#ca6b02',
159:     'nice blue': '#107ab0',
160:     'sapphire': '#2138ab',
161:     'greyish teal': '#719f91',
162:     'orangey yellow': '#fdb915',
163:     'parchment': '#fefcaf',
164:     'straw': '#fcf679',
165:     'very dark brown': '#1d0200',
166:     'terracota': '#cb6843',
167:     'ugly blue': '#31668a',
168:     'clear blue': '#247afd',
169:     'creme': '#ffffb6',
170:     'foam green': '#90fda9',
171:     'grey/green': '#86a17d',
172:     'light gold': '#fddc5c',
173:     'seafoam blue': '#78d1b6',
174:     'topaz': '#13bbaf',
175:     'violet pink': '#fb5ffc',
176:     'wintergreen': '#20f986',
177:     'yellow tan': '#ffe36e',
178:     'dark fuchsia': '#9d0759',
179:     'indigo blue': '#3a18b1',
180:     'light yellowish green': '#c2ff89',
181:     'pale magenta': '#d767ad',
182:     'rich purple': '#720058',
183:     'sunflower yellow': '#ffda03',
184:     'green/blue': '#01c08d',
185:     'leather': '#ac7434',
186:     'racing green': '#014600',
187:     'vivid purple': '#9900fa',
188:     'dark royal blue': '#02066f',
189:     'hazel': '#8e7618',
190:     'muted pink': '#d1768f',
191:     'booger green': '#96b403',
192:     'canary': '#fdff63',
193:     'cool grey': '#95a3a6',
194:     'dark taupe': '#7f684e',
195:     'darkish purple': '#751973',
196:     'true green': '#089404',
197:     'coral pink': '#ff6163',
198:     'dark sage': '#598556',
199:     'dark slate blue': '#214761',
200:     'flat blue': '#3c73a8',
201:     'mushroom': '#ba9e88',
202:     'rich blue': '#021bf9',
203:     'dirty purple': '#734a65',
204:     'greenblue': '#23c48b',
205:     'icky green': '#8fae22',
206:     'light khaki': '#e6f2a2',
207:     'warm blue': '#4b57db',
208:     'dark hot pink': '#d90166',
209:     'deep sea blue': '#015482',
210:     'carmine': '#9d0216',
211:     'dark yellow green': '#728f02',
212:     'pale peach': '#ffe5ad',
213:     'plum purple': '#4e0550',
214:     'golden rod': '#f9bc08',
215:     'neon red': '#ff073a',
216:     'old pink': '#c77986',
217:     'very pale blue': '#d6fffe',
218:     'blood orange': '#fe4b03',
219:     'grapefruit': '#fd5956',
220:     'sand yellow': '#fce166',
221:     'clay brown': '#b2713d',
222:     'dark blue grey': '#1f3b4d',
223:     'flat green': '#699d4c',
224:     'light green blue': '#56fca2',
225:     'warm pink': '#fb5581',
226:     'dodger blue': '#3e82fc',
227:     'gross green': '#a0bf16',
228:     'ice': '#d6fffa',
229:     'metallic blue': '#4f738e',
230:     'pale salmon': '#ffb19a',
231:     'sap green': '#5c8b15',
232:     'algae': '#54ac68',
233:     'bluey grey': '#89a0b0',
234:     'greeny grey': '#7ea07a',
235:     'highlighter green': '#1bfc06',
236:     'light light blue': '#cafffb',
237:     'light mint': '#b6ffbb',
238:     'raw umber': '#a75e09',
239:     'vivid blue': '#152eff',
240:     'deep lavender': '#8d5eb7',
241:     'dull teal': '#5f9e8f',
242:     'light greenish blue': '#63f7b4',
243:     'mud green': '#606602',
244:     'pinky': '#fc86aa',
245:     'red wine': '#8c0034',
246:     'shit green': '#758000',
247:     'tan brown': '#ab7e4c',
248:     'darkblue': '#030764',
249:     'rosa': '#fe86a4',
250:     'lipstick': '#d5174e',
251:     'pale mauve': '#fed0fc',
252:     'claret': '#680018',
253:     'dandelion': '#fedf08',
254:     'orangered': '#fe420f',
255:     'poop green': '#6f7c00',
256:     'ruby': '#ca0147',
257:     'dark': '#1b2431',
258:     'greenish turquoise': '#00fbb0',
259:     'pastel red': '#db5856',
260:     'piss yellow': '#ddd618',
261:     'bright cyan': '#41fdfe',
262:     'dark coral': '#cf524e',
263:     'algae green': '#21c36f',
264:     'darkish red': '#a90308',
265:     'reddy brown': '#6e1005',
266:     'blush pink': '#fe828c',
267:     'camouflage green': '#4b6113',
268:     'lawn green': '#4da409',
269:     'putty': '#beae8a',
270:     'vibrant blue': '#0339f8',
271:     'dark sand': '#a88f59',
272:     'purple/blue': '#5d21d0',
273:     'saffron': '#feb209',
274:     'twilight': '#4e518b',
275:     'warm brown': '#964e02',
276:     'bluegrey': '#85a3b2',
277:     'bubble gum pink': '#ff69af',
278:     'duck egg blue': '#c3fbf4',
279:     'greenish cyan': '#2afeb7',
280:     'petrol': '#005f6a',
281:     'royal': '#0c1793',
282:     'butter': '#ffff81',
283:     'dusty orange': '#f0833a',
284:     'off yellow': '#f1f33f',
285:     'pale olive green': '#b1d27b',
286:     'orangish': '#fc824a',
287:     'leaf': '#71aa34',
288:     'light blue grey': '#b7c9e2',
289:     'dried blood': '#4b0101',
290:     'lightish purple': '#a552e6',
291:     'rusty red': '#af2f0d',
292:     'lavender blue': '#8b88f8',
293:     'light grass green': '#9af764',
294:     'light mint green': '#a6fbb2',
295:     'sunflower': '#ffc512',
296:     'velvet': '#750851',
297:     'brick orange': '#c14a09',
298:     'lightish red': '#fe2f4a',
299:     'pure blue': '#0203e2',
300:     'twilight blue': '#0a437a',
301:     'violet red': '#a50055',
302:     'yellowy brown': '#ae8b0c',
303:     'carnation': '#fd798f',
304:     'muddy yellow': '#bfac05',
305:     'dark seafoam green': '#3eaf76',
306:     'deep rose': '#c74767',
307:     'dusty red': '#b9484e',
308:     'grey/blue': '#647d8e',
309:     'lemon lime': '#bffe28',
310:     'purple/pink': '#d725de',
311:     'brown yellow': '#b29705',
312:     'purple brown': '#673a3f',
313:     'wisteria': '#a87dc2',
314:     'banana yellow': '#fafe4b',
315:     'lipstick red': '#c0022f',
316:     'water blue': '#0e87cc',
317:     'brown grey': '#8d8468',
318:     'vibrant purple': '#ad03de',
319:     'baby green': '#8cff9e',
320:     'barf green': '#94ac02',
321:     'eggshell blue': '#c4fff7',
322:     'sandy yellow': '#fdee73',
323:     'cool green': '#33b864',
324:     'pale': '#fff9d0',
325:     'blue/grey': '#758da3',
326:     'hot magenta': '#f504c9',
327:     'greyblue': '#77a1b5',
328:     'purpley': '#8756e4',
329:     'baby shit green': '#889717',
330:     'brownish pink': '#c27e79',
331:     'dark aquamarine': '#017371',
332:     'diarrhea': '#9f8303',
333:     'light mustard': '#f7d560',
334:     'pale sky blue': '#bdf6fe',
335:     'turtle green': '#75b84f',
336:     'bright olive': '#9cbb04',
337:     'dark grey blue': '#29465b',
338:     'greeny brown': '#696006',
339:     'lemon green': '#adf802',
340:     'light periwinkle': '#c1c6fc',
341:     'seaweed green': '#35ad6b',
342:     'sunshine yellow': '#fffd37',
343:     'ugly purple': '#a442a0',
344:     'medium pink': '#f36196',
345:     'puke brown': '#947706',
346:     'very light pink': '#fff4f2',
347:     'viridian': '#1e9167',
348:     'bile': '#b5c306',
349:     'faded yellow': '#feff7f',
350:     'very pale green': '#cffdbc',
351:     'vibrant green': '#0add08',
352:     'bright lime': '#87fd05',
353:     'spearmint': '#1ef876',
354:     'light aquamarine': '#7bfdc7',
355:     'light sage': '#bcecac',
356:     'yellowgreen': '#bbf90f',
357:     'baby poo': '#ab9004',
358:     'dark seafoam': '#1fb57a',
359:     'deep teal': '#00555a',
360:     'heather': '#a484ac',
361:     'rust orange': '#c45508',
362:     'dirty blue': '#3f829d',
363:     'fern green': '#548d44',
364:     'bright lilac': '#c95efb',
365:     'weird green': '#3ae57f',
366:     'peacock blue': '#016795',
367:     'avocado green': '#87a922',
368:     'faded orange': '#f0944d',
369:     'grape purple': '#5d1451',
370:     'hot green': '#25ff29',
371:     'lime yellow': '#d0fe1d',
372:     'mango': '#ffa62b',
373:     'shamrock': '#01b44c',
374:     'bubblegum': '#ff6cb5',
375:     'purplish brown': '#6b4247',
376:     'vomit yellow': '#c7c10c',
377:     'pale cyan': '#b7fffa',
378:     'key lime': '#aeff6e',
379:     'tomato red': '#ec2d01',
380:     'lightgreen': '#76ff7b',
381:     'merlot': '#730039',
382:     'night blue': '#040348',
383:     'purpleish pink': '#df4ec8',
384:     'apple': '#6ecb3c',
385:     'baby poop green': '#8f9805',
386:     'green apple': '#5edc1f',
387:     'heliotrope': '#d94ff5',
388:     'yellow/green': '#c8fd3d',
389:     'almost black': '#070d0d',
390:     'cool blue': '#4984b8',
391:     'leafy green': '#51b73b',
392:     'mustard brown': '#ac7e04',
393:     'dusk': '#4e5481',
394:     'dull brown': '#876e4b',
395:     'frog green': '#58bc08',
396:     'vivid green': '#2fef10',
397:     'bright light green': '#2dfe54',
398:     'fluro green': '#0aff02',
399:     'kiwi': '#9cef43',
400:     'seaweed': '#18d17b',
401:     'navy green': '#35530a',
402:     'ultramarine blue': '#1805db',
403:     'iris': '#6258c4',
404:     'pastel orange': '#ff964f',
405:     'yellowish orange': '#ffab0f',
406:     'perrywinkle': '#8f8ce7',
407:     'tealish': '#24bca8',
408:     'dark plum': '#3f012c',
409:     'pear': '#cbf85f',
410:     'pinkish orange': '#ff724c',
411:     'midnight purple': '#280137',
412:     'light urple': '#b36ff6',
413:     'dark mint': '#48c072',
414:     'greenish tan': '#bccb7a',
415:     'light burgundy': '#a8415b',
416:     'turquoise blue': '#06b1c4',
417:     'ugly pink': '#cd7584',
418:     'sandy': '#f1da7a',
419:     'electric pink': '#ff0490',
420:     'muted purple': '#805b87',
421:     'mid green': '#50a747',
422:     'greyish': '#a8a495',
423:     'neon yellow': '#cfff04',
424:     'banana': '#ffff7e',
425:     'carnation pink': '#ff7fa7',
426:     'tomato': '#ef4026',
427:     'sea': '#3c9992',
428:     'muddy brown': '#886806',
429:     'turquoise green': '#04f489',
430:     'buff': '#fef69e',
431:     'fawn': '#cfaf7b',
432:     'muted blue': '#3b719f',
433:     'pale rose': '#fdc1c5',
434:     'dark mint green': '#20c073',
435:     'amethyst': '#9b5fc0',
436:     'blue/green': '#0f9b8e',
437:     'chestnut': '#742802',
438:     'sick green': '#9db92c',
439:     'pea': '#a4bf20',
440:     'rusty orange': '#cd5909',
441:     'stone': '#ada587',
442:     'rose red': '#be013c',
443:     'pale aqua': '#b8ffeb',
444:     'deep orange': '#dc4d01',
445:     'earth': '#a2653e',
446:     'mossy green': '#638b27',
447:     'grassy green': '#419c03',
448:     'pale lime green': '#b1ff65',
449:     'light grey blue': '#9dbcd4',
450:     'pale grey': '#fdfdfe',
451:     'asparagus': '#77ab56',
452:     'blueberry': '#464196',
453:     'purple red': '#990147',
454:     'pale lime': '#befd73',
455:     'greenish teal': '#32bf84',
456:     'caramel': '#af6f09',
457:     'deep magenta': '#a0025c',
458:     'light peach': '#ffd8b1',
459:     'milk chocolate': '#7f4e1e',
460:     'ocher': '#bf9b0c',
461:     'off green': '#6ba353',
462:     'purply pink': '#f075e6',
463:     'lightblue': '#7bc8f6',
464:     'dusky blue': '#475f94',
465:     'golden': '#f5bf03',
466:     'light beige': '#fffeb6',
467:     'butter yellow': '#fffd74',
468:     'dusky purple': '#895b7b',
469:     'french blue': '#436bad',
470:     'ugly yellow': '#d0c101',
471:     'greeny yellow': '#c6f808',
472:     'orangish red': '#f43605',
473:     'shamrock green': '#02c14d',
474:     'orangish brown': '#b25f03',
475:     'tree green': '#2a7e19',
476:     'deep violet': '#490648',
477:     'gunmetal': '#536267',
478:     'blue/purple': '#5a06ef',
479:     'cherry': '#cf0234',
480:     'sandy brown': '#c4a661',
481:     'warm grey': '#978a84',
482:     'dark indigo': '#1f0954',
483:     'midnight': '#03012d',
484:     'bluey green': '#2bb179',
485:     'grey pink': '#c3909b',
486:     'soft purple': '#a66fb5',
487:     'blood': '#770001',
488:     'brown red': '#922b05',
489:     'medium grey': '#7d7f7c',
490:     'berry': '#990f4b',
491:     'poo': '#8f7303',
492:     'purpley pink': '#c83cb9',
493:     'light salmon': '#fea993',
494:     'snot': '#acbb0d',
495:     'easter purple': '#c071fe',
496:     'light yellow green': '#ccfd7f',
497:     'dark navy blue': '#00022e',
498:     'drab': '#828344',
499:     'light rose': '#ffc5cb',
500:     'rouge': '#ab1239',
501:     'purplish red': '#b0054b',
502:     'slime green': '#99cc04',
503:     'baby poop': '#937c00',
504:     'irish green': '#019529',
505:     'pink/purple': '#ef1de7',
506:     'dark navy': '#000435',
507:     'greeny blue': '#42b395',
508:     'light plum': '#9d5783',
509:     'pinkish grey': '#c8aca9',
510:     'dirty orange': '#c87606',
511:     'rust red': '#aa2704',
512:     'pale lilac': '#e4cbff',
513:     'orangey red': '#fa4224',
514:     'primary blue': '#0804f9',
515:     'kermit green': '#5cb200',
516:     'brownish purple': '#76424e',
517:     'murky green': '#6c7a0e',
518:     'wheat': '#fbdd7e',
519:     'very dark purple': '#2a0134',
520:     'bottle green': '#044a05',
521:     'watermelon': '#fd4659',
522:     'deep sky blue': '#0d75f8',
523:     'fire engine red': '#fe0002',
524:     'yellow ochre': '#cb9d06',
525:     'pumpkin orange': '#fb7d07',
526:     'pale olive': '#b9cc81',
527:     'light lilac': '#edc8ff',
528:     'lightish green': '#61e160',
529:     'carolina blue': '#8ab8fe',
530:     'mulberry': '#920a4e',
531:     'shocking pink': '#fe02a2',
532:     'auburn': '#9a3001',
533:     'bright lime green': '#65fe08',
534:     'celadon': '#befdb7',
535:     'pinkish brown': '#b17261',
536:     'poo brown': '#885f01',
537:     'bright sky blue': '#02ccfe',
538:     'celery': '#c1fd95',
539:     'dirt brown': '#836539',
540:     'strawberry': '#fb2943',
541:     'dark lime': '#84b701',
542:     'copper': '#b66325',
543:     'medium brown': '#7f5112',
544:     'muted green': '#5fa052',
545:     "robin's egg": '#6dedfd',
546:     'bright aqua': '#0bf9ea',
547:     'bright lavender': '#c760ff',
548:     'ivory': '#ffffcb',
549:     'very light purple': '#f6cefc',
550:     'light navy': '#155084',
551:     'pink red': '#f5054f',
552:     'olive brown': '#645403',
553:     'poop brown': '#7a5901',
554:     'mustard green': '#a8b504',
555:     'ocean green': '#3d9973',
556:     'very dark blue': '#000133',
557:     'dusty green': '#76a973',
558:     'light navy blue': '#2e5a88',
559:     'minty green': '#0bf77d',
560:     'adobe': '#bd6c48',
561:     'barney': '#ac1db8',
562:     'jade green': '#2baf6a',
563:     'bright light blue': '#26f7fd',
564:     'light lime': '#aefd6c',
565:     'dark khaki': '#9b8f55',
566:     'orange yellow': '#ffad01',
567:     'ocre': '#c69c04',
568:     'maize': '#f4d054',
569:     'faded pink': '#de9dac',
570:     'british racing green': '#05480d',
571:     'sandstone': '#c9ae74',
572:     'mud brown': '#60460f',
573:     'light sea green': '#98f6b0',
574:     'robin egg blue': '#8af1fe',
575:     'aqua marine': '#2ee8bb',
576:     'dark sea green': '#11875d',
577:     'soft pink': '#fdb0c0',
578:     'orangey brown': '#b16002',
579:     'cherry red': '#f7022a',
580:     'burnt yellow': '#d5ab09',
581:     'brownish grey': '#86775f',
582:     'camel': '#c69f59',
583:     'purplish grey': '#7a687f',
584:     'marine': '#042e60',
585:     'greyish pink': '#c88d94',
586:     'pale turquoise': '#a5fbd5',
587:     'pastel yellow': '#fffe71',
588:     'bluey purple': '#6241c7',
589:     'canary yellow': '#fffe40',
590:     'faded red': '#d3494e',
591:     'sepia': '#985e2b',
592:     'coffee': '#a6814c',
593:     'bright magenta': '#ff08e8',
594:     'mocha': '#9d7651',
595:     'ecru': '#feffca',
596:     'purpleish': '#98568d',
597:     'cranberry': '#9e003a',
598:     'darkish green': '#287c37',
599:     'brown orange': '#b96902',
600:     'dusky rose': '#ba6873',
601:     'melon': '#ff7855',
602:     'sickly green': '#94b21c',
603:     'silver': '#c5c9c7',
604:     'purply blue': '#661aee',
605:     'purpleish blue': '#6140ef',
606:     'hospital green': '#9be5aa',
607:     'shit brown': '#7b5804',
608:     'mid blue': '#276ab3',
609:     'amber': '#feb308',
610:     'easter green': '#8cfd7e',
611:     'soft blue': '#6488ea',
612:     'cerulean blue': '#056eee',
613:     'golden brown': '#b27a01',
614:     'bright turquoise': '#0ffef9',
615:     'red pink': '#fa2a55',
616:     'red purple': '#820747',
617:     'greyish brown': '#7a6a4f',
618:     'vermillion': '#f4320c',
619:     'russet': '#a13905',
620:     'steel grey': '#6f828a',
621:     'lighter purple': '#a55af4',
622:     'bright violet': '#ad0afd',
623:     'prussian blue': '#004577',
624:     'slate green': '#658d6d',
625:     'dirty pink': '#ca7b80',
626:     'dark blue green': '#005249',
627:     'pine': '#2b5d34',
628:     'yellowy green': '#bff128',
629:     'dark gold': '#b59410',
630:     'bluish': '#2976bb',
631:     'darkish blue': '#014182',
632:     'dull red': '#bb3f3f',
633:     'pinky red': '#fc2647',
634:     'bronze': '#a87900',
635:     'pale teal': '#82cbb2',
636:     'military green': '#667c3e',
637:     'barbie pink': '#fe46a5',
638:     'bubblegum pink': '#fe83cc',
639:     'pea soup green': '#94a617',
640:     'dark mustard': '#a88905',
641:     'shit': '#7f5f00',
642:     'medium purple': '#9e43a2',
643:     'very dark green': '#062e03',
644:     'dirt': '#8a6e45',
645:     'dusky pink': '#cc7a8b',
646:     'red violet': '#9e0168',
647:     'lemon yellow': '#fdff38',
648:     'pistachio': '#c0fa8b',
649:     'dull yellow': '#eedc5b',
650:     'dark lime green': '#7ebd01',
651:     'denim blue': '#3b5b92',
652:     'teal blue': '#01889f',
653:     'lightish blue': '#3d7afd',
654:     'purpley blue': '#5f34e7',
655:     'light indigo': '#6d5acf',
656:     'swamp green': '#748500',
657:     'brown green': '#706c11',
658:     'dark maroon': '#3c0008',
659:     'hot purple': '#cb00f5',
660:     'dark forest green': '#002d04',
661:     'faded blue': '#658cbb',
662:     'drab green': '#749551',
663:     'light lime green': '#b9ff66',
664:     'snot green': '#9dc100',
665:     'yellowish': '#faee66',
666:     'light blue green': '#7efbb3',
667:     'bordeaux': '#7b002c',
668:     'light mauve': '#c292a1',
669:     'ocean': '#017b92',
670:     'marigold': '#fcc006',
671:     'muddy green': '#657432',
672:     'dull orange': '#d8863b',
673:     'steel': '#738595',
674:     'electric purple': '#aa23ff',
675:     'fluorescent green': '#08ff08',
676:     'yellowish brown': '#9b7a01',
677:     'blush': '#f29e8e',
678:     'soft green': '#6fc276',
679:     'bright orange': '#ff5b00',
680:     'lemon': '#fdff52',
681:     'purple grey': '#866f85',
682:     'acid green': '#8ffe09',
683:     'pale lavender': '#eecffe',
684:     'violet blue': '#510ac9',
685:     'light forest green': '#4f9153',
686:     'burnt red': '#9f2305',
687:     'khaki green': '#728639',
688:     'cerise': '#de0c62',
689:     'faded purple': '#916e99',
690:     'apricot': '#ffb16d',
691:     'dark olive green': '#3c4d03',
692:     'grey brown': '#7f7053',
693:     'green grey': '#77926f',
694:     'true blue': '#010fcc',
695:     'pale violet': '#ceaefa',
696:     'periwinkle blue': '#8f99fb',
697:     'light sky blue': '#c6fcff',
698:     'blurple': '#5539cc',
699:     'green brown': '#544e03',
700:     'bluegreen': '#017a79',
701:     'bright teal': '#01f9c6',
702:     'brownish yellow': '#c9b003',
703:     'pea soup': '#929901',
704:     'forest': '#0b5509',
705:     'barney purple': '#a00498',
706:     'ultramarine': '#2000b1',
707:     'purplish': '#94568c',
708:     'puke yellow': '#c2be0e',
709:     'bluish grey': '#748b97',
710:     'dark periwinkle': '#665fd1',
711:     'dark lilac': '#9c6da5',
712:     'reddish': '#c44240',
713:     'light maroon': '#a24857',
714:     'dusty purple': '#825f87',
715:     'terra cotta': '#c9643b',
716:     'avocado': '#90b134',
717:     'marine blue': '#01386a',
718:     'teal green': '#25a36f',
719:     'slate grey': '#59656d',
720:     'lighter green': '#75fd63',
721:     'electric green': '#21fc0d',
722:     'dusty blue': '#5a86ad',
723:     'golden yellow': '#fec615',
724:     'bright yellow': '#fffd01',
725:     'light lavender': '#dfc5fe',
726:     'umber': '#b26400',
727:     'poop': '#7f5e00',
728:     'dark peach': '#de7e5d',
729:     'jungle green': '#048243',
730:     'eggshell': '#ffffd4',
731:     'denim': '#3b638c',
732:     'yellow brown': '#b79400',
733:     'dull purple': '#84597e',
734:     'chocolate brown': '#411900',
735:     'wine red': '#7b0323',
736:     'neon blue': '#04d9ff',
737:     'dirty green': '#667e2c',
738:     'light tan': '#fbeeac',
739:     'ice blue': '#d7fffe',
740:     'cadet blue': '#4e7496',
741:     'dark mauve': '#874c62',
742:     'very light blue': '#d5ffff',
743:     'grey purple': '#826d8c',
744:     'pastel pink': '#ffbacd',
745:     'very light green': '#d1ffbd',
746:     'dark sky blue': '#448ee4',
747:     'evergreen': '#05472a',
748:     'dull pink': '#d5869d',
749:     'aubergine': '#3d0734',
750:     'mahogany': '#4a0100',
751:     'reddish orange': '#f8481c',
752:     'deep green': '#02590f',
753:     'vomit green': '#89a203',
754:     'purple pink': '#e03fd8',
755:     'dusty pink': '#d58a94',
756:     'faded green': '#7bb274',
757:     'camo green': '#526525',
758:     'pinky purple': '#c94cbe',
759:     'pink purple': '#db4bda',
760:     'brownish red': '#9e3623',
761:     'dark rose': '#b5485d',
762:     'mud': '#735c12',
763:     'brownish': '#9c6d57',
764:     'emerald green': '#028f1e',
765:     'pale brown': '#b1916e',
766:     'dull blue': '#49759c',
767:     'burnt umber': '#a0450e',
768:     'medium green': '#39ad48',
769:     'clay': '#b66a50',
770:     'light aqua': '#8cffdb',
771:     'light olive green': '#a4be5c',
772:     'brownish orange': '#cb7723',
773:     'dark aqua': '#05696b',
774:     'purplish pink': '#ce5dae',
775:     'dark salmon': '#c85a53',
776:     'greenish grey': '#96ae8d',
777:     'jade': '#1fa774',
778:     'ugly green': '#7a9703',
779:     'dark beige': '#ac9362',
780:     'emerald': '#01a049',
781:     'pale red': '#d9544d',
782:     'light magenta': '#fa5ff7',
783:     'sky': '#82cafc',
784:     'light cyan': '#acfffc',
785:     'yellow orange': '#fcb001',
786:     'reddish purple': '#910951',
787:     'reddish pink': '#fe2c54',
788:     'orchid': '#c875c4',
789:     'dirty yellow': '#cdc50a',
790:     'orange red': '#fd411e',
791:     'deep red': '#9a0200',
792:     'orange brown': '#be6400',
793:     'cobalt blue': '#030aa7',
794:     'neon pink': '#fe019a',
795:     'rose pink': '#f7879a',
796:     'greyish purple': '#887191',
797:     'raspberry': '#b00149',
798:     'aqua green': '#12e193',
799:     'salmon pink': '#fe7b7c',
800:     'tangerine': '#ff9408',
801:     'brownish green': '#6a6e09',
802:     'red brown': '#8b2e16',
803:     'greenish brown': '#696112',
804:     'pumpkin': '#e17701',
805:     'pine green': '#0a481e',
806:     'charcoal': '#343837',
807:     'baby pink': '#ffb7ce',
808:     'cornflower': '#6a79f7',
809:     'blue violet': '#5d06e9',
810:     'chocolate': '#3d1c02',
811:     'greyish green': '#82a67d',
812:     'scarlet': '#be0119',
813:     'green yellow': '#c9ff27',
814:     'dark olive': '#373e02',
815:     'sienna': '#a9561e',
816:     'pastel purple': '#caa0ff',
817:     'terracotta': '#ca6641',
818:     'aqua blue': '#02d8e9',
819:     'sage green': '#88b378',
820:     'blood red': '#980002',
821:     'deep pink': '#cb0162',
822:     'grass': '#5cac2d',
823:     'moss': '#769958',
824:     'pastel blue': '#a2bffe',
825:     'bluish green': '#10a674',
826:     'green blue': '#06b48b',
827:     'dark tan': '#af884a',
828:     'greenish blue': '#0b8b87',
829:     'pale orange': '#ffa756',
830:     'vomit': '#a2a415',
831:     'forrest green': '#154406',
832:     'dark lavender': '#856798',
833:     'dark violet': '#34013f',
834:     'purple blue': '#632de9',
835:     'dark cyan': '#0a888a',
836:     'olive drab': '#6f7632',
837:     'pinkish': '#d46a7e',
838:     'cobalt': '#1e488f',
839:     'neon purple': '#bc13fe',
840:     'light turquoise': '#7ef4cc',
841:     'apple green': '#76cd26',
842:     'dull green': '#74a662',
843:     'wine': '#80013f',
844:     'powder blue': '#b1d1fc',
845:     'off white': '#ffffe4',
846:     'electric blue': '#0652ff',
847:     'dark turquoise': '#045c5a',
848:     'blue purple': '#5729ce',
849:     'azure': '#069af3',
850:     'bright red': '#ff000d',
851:     'pinkish red': '#f10c45',
852:     'cornflower blue': '#5170d7',
853:     'light olive': '#acbf69',
854:     'grape': '#6c3461',
855:     'greyish blue': '#5e819d',
856:     'purplish blue': '#601ef9',
857:     'yellowish green': '#b0dd16',
858:     'greenish yellow': '#cdfd02',
859:     'medium blue': '#2c6fbb',
860:     'dusty rose': '#c0737a',
861:     'light violet': '#d6b4fc',
862:     'midnight blue': '#020035',
863:     'bluish purple': '#703be7',
864:     'red orange': '#fd3c06',
865:     'dark magenta': '#960056',
866:     'greenish': '#40a368',
867:     'ocean blue': '#03719c',
868:     'coral': '#fc5a50',
869:     'cream': '#ffffc2',
870:     'reddish brown': '#7f2b0a',
871:     'burnt sienna': '#b04e0f',
872:     'brick': '#a03623',
873:     'sage': '#87ae73',
874:     'grey green': '#789b73',
875:     'white': '#ffffff',
876:     "robin's egg blue": '#98eff9',
877:     'moss green': '#658b38',
878:     'steel blue': '#5a7d9a',
879:     'eggplant': '#380835',
880:     'light yellow': '#fffe7a',
881:     'leaf green': '#5ca904',
882:     'light grey': '#d8dcd6',
883:     'puke': '#a5a502',
884:     'pinkish purple': '#d648d7',
885:     'sea blue': '#047495',
886:     'pale purple': '#b790d4',
887:     'slate blue': '#5b7c99',
888:     'blue grey': '#607c8e',
889:     'hunter green': '#0b4008',
890:     'fuchsia': '#ed0dd9',
891:     'crimson': '#8c000f',
892:     'pale yellow': '#ffff84',
893:     'ochre': '#bf9005',
894:     'mustard yellow': '#d2bd0a',
895:     'light red': '#ff474c',
896:     'cerulean': '#0485d1',
897:     'pale pink': '#ffcfdc',
898:     'deep blue': '#040273',
899:     'rust': '#a83c09',
900:     'light teal': '#90e4c1',
901:     'slate': '#516572',
902:     'goldenrod': '#fac205',
903:     'dark yellow': '#d5b60a',
904:     'dark grey': '#363737',
905:     'army green': '#4b5d16',
906:     'grey blue': '#6b8ba4',
907:     'seafoam': '#80f9ad',
908:     'puce': '#a57e52',
909:     'spring green': '#a9f971',
910:     'dark orange': '#c65102',
911:     'sand': '#e2ca76',
912:     'pastel green': '#b0ff9d',
913:     'mint': '#9ffeb0',
914:     'light orange': '#fdaa48',
915:     'bright pink': '#fe01b1',
916:     'chartreuse': '#c1f80a',
917:     'deep purple': '#36013f',
918:     'dark brown': '#341c02',
919:     'taupe': '#b9a281',
920:     'pea green': '#8eab12',
921:     'puke green': '#9aae07',
922:     'kelly green': '#02ab2e',
923:     'seafoam green': '#7af9ab',
924:     'blue green': '#137e6d',
925:     'khaki': '#aaa662',
926:     'burgundy': '#610023',
927:     'dark teal': '#014d4e',
928:     'brick red': '#8f1402',
929:     'royal purple': '#4b006e',
930:     'plum': '#580f41',
931:     'mint green': '#8fff9f',
932:     'gold': '#dbb40c',
933:     'baby blue': '#a2cffe',
934:     'yellow green': '#c0fb2d',
935:     'bright purple': '#be03fd',
936:     'dark red': '#840000',
937:     'pale blue': '#d0fefe',
938:     'grass green': '#3f9b0b',
939:     'navy': '#01153e',
940:     'aquamarine': '#04d8b2',
941:     'burnt orange': '#c04e01',
942:     'neon green': '#0cff0c',
943:     'bright blue': '#0165fc',
944:     'rose': '#cf6275',
945:     'light pink': '#ffd1df',
946:     'mustard': '#ceb301',
947:     'indigo': '#380282',
948:     'lime': '#aaff32',
949:     'sea green': '#53fca1',
950:     'periwinkle': '#8e82fe',
951:     'dark pink': '#cb416b',
952:     'olive green': '#677a04',
953:     'peach': '#ffb07c',
954:     'pale green': '#c7fdb5',
955:     'light brown': '#ad8150',
956:     'hot pink': '#ff028d',
957:     'black': '#000000',
958:     'lilac': '#cea2fd',
959:     'navy blue': '#001146',
960:     'royal blue': '#0504aa',
961:     'beige': '#e6daa6',
962:     'salmon': '#ff796c',
963:     'olive': '#6e750e',
964:     'maroon': '#650021',
965:     'bright green': '#01ff07',
966:     'dark purple': '#35063e',
967:     'mauve': '#ae7181',
968:     'forest green': '#06470c',
969:     'aqua': '#13eac9',
970:     'cyan': '#00ffff',
971:     'tan': '#d1b26f',
972:     'dark blue': '#00035b',
973:     'lavender': '#c79fef',
974:     'turquoise': '#06c2ac',
975:     'dark green': '#033500',
976:     'violet': '#9a0eea',
977:     'light purple': '#bf77f6',
978:     'lime green': '#89fe05',
979:     'grey': '#929591',
980:     'sky blue': '#75bbfd',
981:     'yellow': '#ffff14',
982:     'magenta': '#c20078',
983:     'light green': '#96f97b',
984:     'orange': '#f97306',
985:     'teal': '#029386',
986:     'light blue': '#95d0fc',
987:     'red': '#e50000',
988:     'brown': '#653700',
989:     'pink': '#ff81c0',
990:     'blue': '#0343df',
991:     'green': '#15b01a',
992:     'purple': '#7e1e9c'}
993: 
994: # Normalize name to "xkcd:<name>" to avoid name collisions.
995: XKCD_COLORS = {'xkcd:' + name: value for name, value in XKCD_COLORS.items()}
996: 
997: 
998: # https://drafts.csswg.org/css-color-4/#named-colors
999: CSS4_COLORS = {
1000:     'aliceblue':            '#F0F8FF',
1001:     'antiquewhite':         '#FAEBD7',
1002:     'aqua':                 '#00FFFF',
1003:     'aquamarine':           '#7FFFD4',
1004:     'azure':                '#F0FFFF',
1005:     'beige':                '#F5F5DC',
1006:     'bisque':               '#FFE4C4',
1007:     'black':                '#000000',
1008:     'blanchedalmond':       '#FFEBCD',
1009:     'blue':                 '#0000FF',
1010:     'blueviolet':           '#8A2BE2',
1011:     'brown':                '#A52A2A',
1012:     'burlywood':            '#DEB887',
1013:     'cadetblue':            '#5F9EA0',
1014:     'chartreuse':           '#7FFF00',
1015:     'chocolate':            '#D2691E',
1016:     'coral':                '#FF7F50',
1017:     'cornflowerblue':       '#6495ED',
1018:     'cornsilk':             '#FFF8DC',
1019:     'crimson':              '#DC143C',
1020:     'cyan':                 '#00FFFF',
1021:     'darkblue':             '#00008B',
1022:     'darkcyan':             '#008B8B',
1023:     'darkgoldenrod':        '#B8860B',
1024:     'darkgray':             '#A9A9A9',
1025:     'darkgreen':            '#006400',
1026:     'darkgrey':             '#A9A9A9',
1027:     'darkkhaki':            '#BDB76B',
1028:     'darkmagenta':          '#8B008B',
1029:     'darkolivegreen':       '#556B2F',
1030:     'darkorange':           '#FF8C00',
1031:     'darkorchid':           '#9932CC',
1032:     'darkred':              '#8B0000',
1033:     'darksalmon':           '#E9967A',
1034:     'darkseagreen':         '#8FBC8F',
1035:     'darkslateblue':        '#483D8B',
1036:     'darkslategray':        '#2F4F4F',
1037:     'darkslategrey':        '#2F4F4F',
1038:     'darkturquoise':        '#00CED1',
1039:     'darkviolet':           '#9400D3',
1040:     'deeppink':             '#FF1493',
1041:     'deepskyblue':          '#00BFFF',
1042:     'dimgray':              '#696969',
1043:     'dimgrey':              '#696969',
1044:     'dodgerblue':           '#1E90FF',
1045:     'firebrick':            '#B22222',
1046:     'floralwhite':          '#FFFAF0',
1047:     'forestgreen':          '#228B22',
1048:     'fuchsia':              '#FF00FF',
1049:     'gainsboro':            '#DCDCDC',
1050:     'ghostwhite':           '#F8F8FF',
1051:     'gold':                 '#FFD700',
1052:     'goldenrod':            '#DAA520',
1053:     'gray':                 '#808080',
1054:     'green':                '#008000',
1055:     'greenyellow':          '#ADFF2F',
1056:     'grey':                 '#808080',
1057:     'honeydew':             '#F0FFF0',
1058:     'hotpink':              '#FF69B4',
1059:     'indianred':            '#CD5C5C',
1060:     'indigo':               '#4B0082',
1061:     'ivory':                '#FFFFF0',
1062:     'khaki':                '#F0E68C',
1063:     'lavender':             '#E6E6FA',
1064:     'lavenderblush':        '#FFF0F5',
1065:     'lawngreen':            '#7CFC00',
1066:     'lemonchiffon':         '#FFFACD',
1067:     'lightblue':            '#ADD8E6',
1068:     'lightcoral':           '#F08080',
1069:     'lightcyan':            '#E0FFFF',
1070:     'lightgoldenrodyellow': '#FAFAD2',
1071:     'lightgray':            '#D3D3D3',
1072:     'lightgreen':           '#90EE90',
1073:     'lightgrey':            '#D3D3D3',
1074:     'lightpink':            '#FFB6C1',
1075:     'lightsalmon':          '#FFA07A',
1076:     'lightseagreen':        '#20B2AA',
1077:     'lightskyblue':         '#87CEFA',
1078:     'lightslategray':       '#778899',
1079:     'lightslategrey':       '#778899',
1080:     'lightsteelblue':       '#B0C4DE',
1081:     'lightyellow':          '#FFFFE0',
1082:     'lime':                 '#00FF00',
1083:     'limegreen':            '#32CD32',
1084:     'linen':                '#FAF0E6',
1085:     'magenta':              '#FF00FF',
1086:     'maroon':               '#800000',
1087:     'mediumaquamarine':     '#66CDAA',
1088:     'mediumblue':           '#0000CD',
1089:     'mediumorchid':         '#BA55D3',
1090:     'mediumpurple':         '#9370DB',
1091:     'mediumseagreen':       '#3CB371',
1092:     'mediumslateblue':      '#7B68EE',
1093:     'mediumspringgreen':    '#00FA9A',
1094:     'mediumturquoise':      '#48D1CC',
1095:     'mediumvioletred':      '#C71585',
1096:     'midnightblue':         '#191970',
1097:     'mintcream':            '#F5FFFA',
1098:     'mistyrose':            '#FFE4E1',
1099:     'moccasin':             '#FFE4B5',
1100:     'navajowhite':          '#FFDEAD',
1101:     'navy':                 '#000080',
1102:     'oldlace':              '#FDF5E6',
1103:     'olive':                '#808000',
1104:     'olivedrab':            '#6B8E23',
1105:     'orange':               '#FFA500',
1106:     'orangered':            '#FF4500',
1107:     'orchid':               '#DA70D6',
1108:     'palegoldenrod':        '#EEE8AA',
1109:     'palegreen':            '#98FB98',
1110:     'paleturquoise':        '#AFEEEE',
1111:     'palevioletred':        '#DB7093',
1112:     'papayawhip':           '#FFEFD5',
1113:     'peachpuff':            '#FFDAB9',
1114:     'peru':                 '#CD853F',
1115:     'pink':                 '#FFC0CB',
1116:     'plum':                 '#DDA0DD',
1117:     'powderblue':           '#B0E0E6',
1118:     'purple':               '#800080',
1119:     'rebeccapurple':        '#663399',
1120:     'red':                  '#FF0000',
1121:     'rosybrown':            '#BC8F8F',
1122:     'royalblue':            '#4169E1',
1123:     'saddlebrown':          '#8B4513',
1124:     'salmon':               '#FA8072',
1125:     'sandybrown':           '#F4A460',
1126:     'seagreen':             '#2E8B57',
1127:     'seashell':             '#FFF5EE',
1128:     'sienna':               '#A0522D',
1129:     'silver':               '#C0C0C0',
1130:     'skyblue':              '#87CEEB',
1131:     'slateblue':            '#6A5ACD',
1132:     'slategray':            '#708090',
1133:     'slategrey':            '#708090',
1134:     'snow':                 '#FFFAFA',
1135:     'springgreen':          '#00FF7F',
1136:     'steelblue':            '#4682B4',
1137:     'tan':                  '#D2B48C',
1138:     'teal':                 '#008080',
1139:     'thistle':              '#D8BFD8',
1140:     'tomato':               '#FF6347',
1141:     'turquoise':            '#40E0D0',
1142:     'violet':               '#EE82EE',
1143:     'wheat':                '#F5DEB3',
1144:     'white':                '#FFFFFF',
1145:     'whitesmoke':           '#F5F5F5',
1146:     'yellow':               '#FFFF00',
1147:     'yellowgreen':          '#9ACD32'}
1148: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from collections import OrderedDict' statement (line 3)
try:
    from collections import OrderedDict

except:
    OrderedDict = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'collections', None, module_type_store, ['OrderedDict'], [OrderedDict])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import six' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_179468 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six')

if (type(import_179468) is not StypyTypeError):

    if (import_179468 != 'pyd_module'):
        __import__(import_179468)
        sys_modules_179469 = sys.modules[import_179468]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', sys_modules_179469.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'six', import_179468)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')


# Assigning a Dict to a Name (line 7):

# Obtaining an instance of the builtin type 'dict' (line 7)
dict_179470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 14), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 7)
# Adding element type (key, value) (line 7)
unicode_179471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 4), 'unicode', u'b')

# Obtaining an instance of the builtin type 'tuple' (line 8)
tuple_179472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 10), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 8)
# Adding element type (line 8)
int_179473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 10), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), tuple_179472, int_179473)
# Adding element type (line 8)
int_179474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), tuple_179472, int_179474)
# Adding element type (line 8)
int_179475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), tuple_179472, int_179475)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 14), dict_179470, (unicode_179471, tuple_179472))
# Adding element type (key, value) (line 7)
unicode_179476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 4), 'unicode', u'g')

# Obtaining an instance of the builtin type 'tuple' (line 9)
tuple_179477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 10), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 9)
# Adding element type (line 9)
int_179478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 10), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), tuple_179477, int_179478)
# Adding element type (line 9)
float_179479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 13), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), tuple_179477, float_179479)
# Adding element type (line 9)
int_179480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), tuple_179477, int_179480)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 14), dict_179470, (unicode_179476, tuple_179477))
# Adding element type (key, value) (line 7)
unicode_179481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 4), 'unicode', u'r')

# Obtaining an instance of the builtin type 'tuple' (line 10)
tuple_179482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 10), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 10)
# Adding element type (line 10)
int_179483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 10), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), tuple_179482, int_179483)
# Adding element type (line 10)
int_179484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), tuple_179482, int_179484)
# Adding element type (line 10)
int_179485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), tuple_179482, int_179485)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 14), dict_179470, (unicode_179481, tuple_179482))
# Adding element type (key, value) (line 7)
unicode_179486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 4), 'unicode', u'c')

# Obtaining an instance of the builtin type 'tuple' (line 11)
tuple_179487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 10), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 11)
# Adding element type (line 11)
int_179488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 10), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 10), tuple_179487, int_179488)
# Adding element type (line 11)
float_179489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 13), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 10), tuple_179487, float_179489)
# Adding element type (line 11)
float_179490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 19), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 10), tuple_179487, float_179490)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 14), dict_179470, (unicode_179486, tuple_179487))
# Adding element type (key, value) (line 7)
unicode_179491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 4), 'unicode', u'm')

# Obtaining an instance of the builtin type 'tuple' (line 12)
tuple_179492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 10), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 12)
# Adding element type (line 12)
float_179493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), tuple_179492, float_179493)
# Adding element type (line 12)
int_179494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), tuple_179492, int_179494)
# Adding element type (line 12)
float_179495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 19), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), tuple_179492, float_179495)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 14), dict_179470, (unicode_179491, tuple_179492))
# Adding element type (key, value) (line 7)
unicode_179496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 4), 'unicode', u'y')

# Obtaining an instance of the builtin type 'tuple' (line 13)
tuple_179497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 10), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 13)
# Adding element type (line 13)
float_179498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), tuple_179497, float_179498)
# Adding element type (line 13)
float_179499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), tuple_179497, float_179499)
# Adding element type (line 13)
int_179500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), tuple_179497, int_179500)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 14), dict_179470, (unicode_179496, tuple_179497))
# Adding element type (key, value) (line 7)
unicode_179501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 4), 'unicode', u'k')

# Obtaining an instance of the builtin type 'tuple' (line 14)
tuple_179502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 10), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 14)
# Adding element type (line 14)
int_179503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 10), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), tuple_179502, int_179503)
# Adding element type (line 14)
int_179504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), tuple_179502, int_179504)
# Adding element type (line 14)
int_179505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), tuple_179502, int_179505)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 14), dict_179470, (unicode_179501, tuple_179502))
# Adding element type (key, value) (line 7)
unicode_179506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 4), 'unicode', u'w')

# Obtaining an instance of the builtin type 'tuple' (line 15)
tuple_179507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 10), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 15)
# Adding element type (line 15)
int_179508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 10), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), tuple_179507, int_179508)
# Adding element type (line 15)
int_179509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), tuple_179507, int_179509)
# Adding element type (line 15)
int_179510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), tuple_179507, int_179510)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 14), dict_179470, (unicode_179506, tuple_179507))

# Assigning a type to the variable 'BASE_COLORS' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'BASE_COLORS', dict_179470)

# Assigning a Tuple to a Name (line 19):

# Obtaining an instance of the builtin type 'tuple' (line 20)
tuple_179511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 4), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 20)
# Adding element type (line 20)

# Obtaining an instance of the builtin type 'tuple' (line 20)
tuple_179512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 20)
# Adding element type (line 20)
unicode_179513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 5), 'unicode', u'blue')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 5), tuple_179512, unicode_179513)
# Adding element type (line 20)
unicode_179514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 13), 'unicode', u'#1f77b4')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 5), tuple_179512, unicode_179514)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 4), tuple_179511, tuple_179512)
# Adding element type (line 20)

# Obtaining an instance of the builtin type 'tuple' (line 21)
tuple_179515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 21)
# Adding element type (line 21)
unicode_179516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 5), 'unicode', u'orange')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 5), tuple_179515, unicode_179516)
# Adding element type (line 21)
unicode_179517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 15), 'unicode', u'#ff7f0e')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 5), tuple_179515, unicode_179517)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 4), tuple_179511, tuple_179515)
# Adding element type (line 20)

# Obtaining an instance of the builtin type 'tuple' (line 22)
tuple_179518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 22)
# Adding element type (line 22)
unicode_179519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 5), 'unicode', u'green')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 5), tuple_179518, unicode_179519)
# Adding element type (line 22)
unicode_179520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 14), 'unicode', u'#2ca02c')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 5), tuple_179518, unicode_179520)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 4), tuple_179511, tuple_179518)
# Adding element type (line 20)

# Obtaining an instance of the builtin type 'tuple' (line 23)
tuple_179521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 23)
# Adding element type (line 23)
unicode_179522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 5), 'unicode', u'red')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 5), tuple_179521, unicode_179522)
# Adding element type (line 23)
unicode_179523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 12), 'unicode', u'#d62728')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 5), tuple_179521, unicode_179523)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 4), tuple_179511, tuple_179521)
# Adding element type (line 20)

# Obtaining an instance of the builtin type 'tuple' (line 24)
tuple_179524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 24)
# Adding element type (line 24)
unicode_179525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 5), 'unicode', u'purple')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 5), tuple_179524, unicode_179525)
# Adding element type (line 24)
unicode_179526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 15), 'unicode', u'#9467bd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 5), tuple_179524, unicode_179526)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 4), tuple_179511, tuple_179524)
# Adding element type (line 20)

# Obtaining an instance of the builtin type 'tuple' (line 25)
tuple_179527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 25)
# Adding element type (line 25)
unicode_179528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 5), 'unicode', u'brown')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 5), tuple_179527, unicode_179528)
# Adding element type (line 25)
unicode_179529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 14), 'unicode', u'#8c564b')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 5), tuple_179527, unicode_179529)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 4), tuple_179511, tuple_179527)
# Adding element type (line 20)

# Obtaining an instance of the builtin type 'tuple' (line 26)
tuple_179530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 26)
# Adding element type (line 26)
unicode_179531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 5), 'unicode', u'pink')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 5), tuple_179530, unicode_179531)
# Adding element type (line 26)
unicode_179532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 13), 'unicode', u'#e377c2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 5), tuple_179530, unicode_179532)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 4), tuple_179511, tuple_179530)
# Adding element type (line 20)

# Obtaining an instance of the builtin type 'tuple' (line 27)
tuple_179533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 27)
# Adding element type (line 27)
unicode_179534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 5), 'unicode', u'gray')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 5), tuple_179533, unicode_179534)
# Adding element type (line 27)
unicode_179535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 13), 'unicode', u'#7f7f7f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 5), tuple_179533, unicode_179535)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 4), tuple_179511, tuple_179533)
# Adding element type (line 20)

# Obtaining an instance of the builtin type 'tuple' (line 28)
tuple_179536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 28)
# Adding element type (line 28)
unicode_179537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 5), 'unicode', u'olive')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 5), tuple_179536, unicode_179537)
# Adding element type (line 28)
unicode_179538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 14), 'unicode', u'#bcbd22')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 5), tuple_179536, unicode_179538)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 4), tuple_179511, tuple_179536)
# Adding element type (line 20)

# Obtaining an instance of the builtin type 'tuple' (line 29)
tuple_179539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 29)
# Adding element type (line 29)
unicode_179540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 5), 'unicode', u'cyan')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 5), tuple_179539, unicode_179540)
# Adding element type (line 29)
unicode_179541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 13), 'unicode', u'#17becf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 5), tuple_179539, unicode_179541)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 4), tuple_179511, tuple_179539)

# Assigning a type to the variable 'TABLEAU_COLORS' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'TABLEAU_COLORS', tuple_179511)

# Assigning a Call to a Name (line 33):

# Call to OrderedDict(...): (line 33)
# Processing the call arguments (line 33)
# Calculating generator expression
module_type_store = module_type_store.open_function_context('list comprehension expression', 34, 4, True)
# Calculating comprehension expression
# Getting the type of 'TABLEAU_COLORS' (line 34)
TABLEAU_COLORS_179548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 46), 'TABLEAU_COLORS', False)
comprehension_179549 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 4), TABLEAU_COLORS_179548)
# Assigning a type to the variable 'name' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 4), comprehension_179549))
# Assigning a type to the variable 'value' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 4), comprehension_179549))

# Obtaining an instance of the builtin type 'tuple' (line 34)
tuple_179543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 34)
# Adding element type (line 34)
unicode_179544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 5), 'unicode', u'tab:')
# Getting the type of 'name' (line 34)
name_179545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 14), 'name', False)
# Applying the binary operator '+' (line 34)
result_add_179546 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 5), '+', unicode_179544, name_179545)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 5), tuple_179543, result_add_179546)
# Adding element type (line 34)
# Getting the type of 'value' (line 34)
value_179547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 20), 'value', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 5), tuple_179543, value_179547)

list_179550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 4), 'list')
# Destroy the current context
module_type_store = module_type_store.close_function_context()
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 4), list_179550, tuple_179543)
# Processing the call keyword arguments (line 33)
kwargs_179551 = {}
# Getting the type of 'OrderedDict' (line 33)
OrderedDict_179542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 17), 'OrderedDict', False)
# Calling OrderedDict(args, kwargs) (line 33)
OrderedDict_call_result_179552 = invoke(stypy.reporting.localization.Localization(__file__, 33, 17), OrderedDict_179542, *[list_179550], **kwargs_179551)

# Assigning a type to the variable 'TABLEAU_COLORS' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'TABLEAU_COLORS', OrderedDict_call_result_179552)

# Assigning a Dict to a Name (line 43):

# Obtaining an instance of the builtin type 'dict' (line 43)
dict_179553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 14), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 43)
# Adding element type (key, value) (line 43)
unicode_179554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 4), 'unicode', u'cloudy blue')
unicode_179555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 19), 'unicode', u'#acc2d9')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179554, unicode_179555))
# Adding element type (key, value) (line 43)
unicode_179556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 4), 'unicode', u'dark pastel green')
unicode_179557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 25), 'unicode', u'#56ae57')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179556, unicode_179557))
# Adding element type (key, value) (line 43)
unicode_179558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 4), 'unicode', u'dust')
unicode_179559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 12), 'unicode', u'#b2996e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179558, unicode_179559))
# Adding element type (key, value) (line 43)
unicode_179560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 4), 'unicode', u'electric lime')
unicode_179561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 21), 'unicode', u'#a8ff04')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179560, unicode_179561))
# Adding element type (key, value) (line 43)
unicode_179562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 4), 'unicode', u'fresh green')
unicode_179563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 19), 'unicode', u'#69d84f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179562, unicode_179563))
# Adding element type (key, value) (line 43)
unicode_179564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 4), 'unicode', u'light eggplant')
unicode_179565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 22), 'unicode', u'#894585')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179564, unicode_179565))
# Adding element type (key, value) (line 43)
unicode_179566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 4), 'unicode', u'nasty green')
unicode_179567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 19), 'unicode', u'#70b23f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179566, unicode_179567))
# Adding element type (key, value) (line 43)
unicode_179568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 4), 'unicode', u'really light blue')
unicode_179569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 25), 'unicode', u'#d4ffff')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179568, unicode_179569))
# Adding element type (key, value) (line 43)
unicode_179570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 4), 'unicode', u'tea')
unicode_179571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 11), 'unicode', u'#65ab7c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179570, unicode_179571))
# Adding element type (key, value) (line 43)
unicode_179572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 4), 'unicode', u'warm purple')
unicode_179573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 19), 'unicode', u'#952e8f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179572, unicode_179573))
# Adding element type (key, value) (line 43)
unicode_179574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 4), 'unicode', u'yellowish tan')
unicode_179575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 21), 'unicode', u'#fcfc81')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179574, unicode_179575))
# Adding element type (key, value) (line 43)
unicode_179576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 4), 'unicode', u'cement')
unicode_179577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 14), 'unicode', u'#a5a391')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179576, unicode_179577))
# Adding element type (key, value) (line 43)
unicode_179578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 4), 'unicode', u'dark grass green')
unicode_179579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 24), 'unicode', u'#388004')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179578, unicode_179579))
# Adding element type (key, value) (line 43)
unicode_179580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 4), 'unicode', u'dusty teal')
unicode_179581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 18), 'unicode', u'#4c9085')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179580, unicode_179581))
# Adding element type (key, value) (line 43)
unicode_179582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 4), 'unicode', u'grey teal')
unicode_179583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 17), 'unicode', u'#5e9b8a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179582, unicode_179583))
# Adding element type (key, value) (line 43)
unicode_179584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 4), 'unicode', u'macaroni and cheese')
unicode_179585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 27), 'unicode', u'#efb435')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179584, unicode_179585))
# Adding element type (key, value) (line 43)
unicode_179586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 4), 'unicode', u'pinkish tan')
unicode_179587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 19), 'unicode', u'#d99b82')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179586, unicode_179587))
# Adding element type (key, value) (line 43)
unicode_179588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 4), 'unicode', u'spruce')
unicode_179589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 14), 'unicode', u'#0a5f38')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179588, unicode_179589))
# Adding element type (key, value) (line 43)
unicode_179590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 4), 'unicode', u'strong blue')
unicode_179591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 19), 'unicode', u'#0c06f7')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179590, unicode_179591))
# Adding element type (key, value) (line 43)
unicode_179592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 4), 'unicode', u'toxic green')
unicode_179593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 19), 'unicode', u'#61de2a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179592, unicode_179593))
# Adding element type (key, value) (line 43)
unicode_179594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 4), 'unicode', u'windows blue')
unicode_179595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 20), 'unicode', u'#3778bf')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179594, unicode_179595))
# Adding element type (key, value) (line 43)
unicode_179596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 4), 'unicode', u'blue blue')
unicode_179597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 17), 'unicode', u'#2242c7')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179596, unicode_179597))
# Adding element type (key, value) (line 43)
unicode_179598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 4), 'unicode', u'blue with a hint of purple')
unicode_179599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 34), 'unicode', u'#533cc6')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179598, unicode_179599))
# Adding element type (key, value) (line 43)
unicode_179600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 4), 'unicode', u'booger')
unicode_179601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 14), 'unicode', u'#9bb53c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179600, unicode_179601))
# Adding element type (key, value) (line 43)
unicode_179602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 4), 'unicode', u'bright sea green')
unicode_179603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 24), 'unicode', u'#05ffa6')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179602, unicode_179603))
# Adding element type (key, value) (line 43)
unicode_179604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 4), 'unicode', u'dark green blue')
unicode_179605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 23), 'unicode', u'#1f6357')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179604, unicode_179605))
# Adding element type (key, value) (line 43)
unicode_179606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 4), 'unicode', u'deep turquoise')
unicode_179607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 22), 'unicode', u'#017374')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179606, unicode_179607))
# Adding element type (key, value) (line 43)
unicode_179608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 4), 'unicode', u'green teal')
unicode_179609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 18), 'unicode', u'#0cb577')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179608, unicode_179609))
# Adding element type (key, value) (line 43)
unicode_179610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 4), 'unicode', u'strong pink')
unicode_179611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 19), 'unicode', u'#ff0789')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179610, unicode_179611))
# Adding element type (key, value) (line 43)
unicode_179612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 4), 'unicode', u'bland')
unicode_179613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 13), 'unicode', u'#afa88b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179612, unicode_179613))
# Adding element type (key, value) (line 43)
unicode_179614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 4), 'unicode', u'deep aqua')
unicode_179615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 17), 'unicode', u'#08787f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179614, unicode_179615))
# Adding element type (key, value) (line 43)
unicode_179616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 4), 'unicode', u'lavender pink')
unicode_179617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 21), 'unicode', u'#dd85d7')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179616, unicode_179617))
# Adding element type (key, value) (line 43)
unicode_179618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 4), 'unicode', u'light moss green')
unicode_179619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 24), 'unicode', u'#a6c875')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179618, unicode_179619))
# Adding element type (key, value) (line 43)
unicode_179620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 4), 'unicode', u'light seafoam green')
unicode_179621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 27), 'unicode', u'#a7ffb5')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179620, unicode_179621))
# Adding element type (key, value) (line 43)
unicode_179622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 4), 'unicode', u'olive yellow')
unicode_179623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 20), 'unicode', u'#c2b709')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179622, unicode_179623))
# Adding element type (key, value) (line 43)
unicode_179624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 4), 'unicode', u'pig pink')
unicode_179625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 16), 'unicode', u'#e78ea5')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179624, unicode_179625))
# Adding element type (key, value) (line 43)
unicode_179626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 4), 'unicode', u'deep lilac')
unicode_179627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 18), 'unicode', u'#966ebd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179626, unicode_179627))
# Adding element type (key, value) (line 43)
unicode_179628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 4), 'unicode', u'desert')
unicode_179629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 14), 'unicode', u'#ccad60')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179628, unicode_179629))
# Adding element type (key, value) (line 43)
unicode_179630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 4), 'unicode', u'dusty lavender')
unicode_179631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 22), 'unicode', u'#ac86a8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179630, unicode_179631))
# Adding element type (key, value) (line 43)
unicode_179632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 4), 'unicode', u'purpley grey')
unicode_179633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 20), 'unicode', u'#947e94')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179632, unicode_179633))
# Adding element type (key, value) (line 43)
unicode_179634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 4), 'unicode', u'purply')
unicode_179635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 14), 'unicode', u'#983fb2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179634, unicode_179635))
# Adding element type (key, value) (line 43)
unicode_179636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 4), 'unicode', u'candy pink')
unicode_179637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 18), 'unicode', u'#ff63e9')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179636, unicode_179637))
# Adding element type (key, value) (line 43)
unicode_179638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 4), 'unicode', u'light pastel green')
unicode_179639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 26), 'unicode', u'#b2fba5')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179638, unicode_179639))
# Adding element type (key, value) (line 43)
unicode_179640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 4), 'unicode', u'boring green')
unicode_179641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 20), 'unicode', u'#63b365')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179640, unicode_179641))
# Adding element type (key, value) (line 43)
unicode_179642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 4), 'unicode', u'kiwi green')
unicode_179643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 18), 'unicode', u'#8ee53f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179642, unicode_179643))
# Adding element type (key, value) (line 43)
unicode_179644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 4), 'unicode', u'light grey green')
unicode_179645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 24), 'unicode', u'#b7e1a1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179644, unicode_179645))
# Adding element type (key, value) (line 43)
unicode_179646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 4), 'unicode', u'orange pink')
unicode_179647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 19), 'unicode', u'#ff6f52')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179646, unicode_179647))
# Adding element type (key, value) (line 43)
unicode_179648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 4), 'unicode', u'tea green')
unicode_179649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 17), 'unicode', u'#bdf8a3')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179648, unicode_179649))
# Adding element type (key, value) (line 43)
unicode_179650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 4), 'unicode', u'very light brown')
unicode_179651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 24), 'unicode', u'#d3b683')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179650, unicode_179651))
# Adding element type (key, value) (line 43)
unicode_179652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 4), 'unicode', u'egg shell')
unicode_179653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 17), 'unicode', u'#fffcc4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179652, unicode_179653))
# Adding element type (key, value) (line 43)
unicode_179654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 4), 'unicode', u'eggplant purple')
unicode_179655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 23), 'unicode', u'#430541')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179654, unicode_179655))
# Adding element type (key, value) (line 43)
unicode_179656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 4), 'unicode', u'powder pink')
unicode_179657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 19), 'unicode', u'#ffb2d0')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179656, unicode_179657))
# Adding element type (key, value) (line 43)
unicode_179658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 4), 'unicode', u'reddish grey')
unicode_179659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 20), 'unicode', u'#997570')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179658, unicode_179659))
# Adding element type (key, value) (line 43)
unicode_179660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 4), 'unicode', u'baby shit brown')
unicode_179661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 23), 'unicode', u'#ad900d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179660, unicode_179661))
# Adding element type (key, value) (line 43)
unicode_179662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 4), 'unicode', u'liliac')
unicode_179663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 14), 'unicode', u'#c48efd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179662, unicode_179663))
# Adding element type (key, value) (line 43)
unicode_179664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 4), 'unicode', u'stormy blue')
unicode_179665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 19), 'unicode', u'#507b9c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179664, unicode_179665))
# Adding element type (key, value) (line 43)
unicode_179666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 4), 'unicode', u'ugly brown')
unicode_179667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 18), 'unicode', u'#7d7103')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179666, unicode_179667))
# Adding element type (key, value) (line 43)
unicode_179668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 4), 'unicode', u'custard')
unicode_179669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 15), 'unicode', u'#fffd78')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179668, unicode_179669))
# Adding element type (key, value) (line 43)
unicode_179670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 4), 'unicode', u'darkish pink')
unicode_179671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 20), 'unicode', u'#da467d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179670, unicode_179671))
# Adding element type (key, value) (line 43)
unicode_179672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 4), 'unicode', u'deep brown')
unicode_179673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 18), 'unicode', u'#410200')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179672, unicode_179673))
# Adding element type (key, value) (line 43)
unicode_179674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 4), 'unicode', u'greenish beige')
unicode_179675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 22), 'unicode', u'#c9d179')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179674, unicode_179675))
# Adding element type (key, value) (line 43)
unicode_179676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 4), 'unicode', u'manilla')
unicode_179677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 15), 'unicode', u'#fffa86')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179676, unicode_179677))
# Adding element type (key, value) (line 43)
unicode_179678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 4), 'unicode', u'off blue')
unicode_179679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 16), 'unicode', u'#5684ae')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179678, unicode_179679))
# Adding element type (key, value) (line 43)
unicode_179680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 4), 'unicode', u'battleship grey')
unicode_179681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 23), 'unicode', u'#6b7c85')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179680, unicode_179681))
# Adding element type (key, value) (line 43)
unicode_179682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 4), 'unicode', u'browny green')
unicode_179683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 20), 'unicode', u'#6f6c0a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179682, unicode_179683))
# Adding element type (key, value) (line 43)
unicode_179684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 4), 'unicode', u'bruise')
unicode_179685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 14), 'unicode', u'#7e4071')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179684, unicode_179685))
# Adding element type (key, value) (line 43)
unicode_179686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 4), 'unicode', u'kelley green')
unicode_179687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 20), 'unicode', u'#009337')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179686, unicode_179687))
# Adding element type (key, value) (line 43)
unicode_179688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 4), 'unicode', u'sickly yellow')
unicode_179689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 21), 'unicode', u'#d0e429')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179688, unicode_179689))
# Adding element type (key, value) (line 43)
unicode_179690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 4), 'unicode', u'sunny yellow')
unicode_179691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 20), 'unicode', u'#fff917')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179690, unicode_179691))
# Adding element type (key, value) (line 43)
unicode_179692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 4), 'unicode', u'azul')
unicode_179693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 12), 'unicode', u'#1d5dec')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179692, unicode_179693))
# Adding element type (key, value) (line 43)
unicode_179694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 4), 'unicode', u'darkgreen')
unicode_179695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 17), 'unicode', u'#054907')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179694, unicode_179695))
# Adding element type (key, value) (line 43)
unicode_179696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 4), 'unicode', u'green/yellow')
unicode_179697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 20), 'unicode', u'#b5ce08')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179696, unicode_179697))
# Adding element type (key, value) (line 43)
unicode_179698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 4), 'unicode', u'lichen')
unicode_179699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 14), 'unicode', u'#8fb67b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179698, unicode_179699))
# Adding element type (key, value) (line 43)
unicode_179700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 4), 'unicode', u'light light green')
unicode_179701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 25), 'unicode', u'#c8ffb0')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179700, unicode_179701))
# Adding element type (key, value) (line 43)
unicode_179702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 4), 'unicode', u'pale gold')
unicode_179703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 17), 'unicode', u'#fdde6c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179702, unicode_179703))
# Adding element type (key, value) (line 43)
unicode_179704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 4), 'unicode', u'sun yellow')
unicode_179705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 18), 'unicode', u'#ffdf22')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179704, unicode_179705))
# Adding element type (key, value) (line 43)
unicode_179706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 4), 'unicode', u'tan green')
unicode_179707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 17), 'unicode', u'#a9be70')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179706, unicode_179707))
# Adding element type (key, value) (line 43)
unicode_179708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 4), 'unicode', u'burple')
unicode_179709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 14), 'unicode', u'#6832e3')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179708, unicode_179709))
# Adding element type (key, value) (line 43)
unicode_179710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 4), 'unicode', u'butterscotch')
unicode_179711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 20), 'unicode', u'#fdb147')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179710, unicode_179711))
# Adding element type (key, value) (line 43)
unicode_179712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 4), 'unicode', u'toupe')
unicode_179713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 13), 'unicode', u'#c7ac7d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179712, unicode_179713))
# Adding element type (key, value) (line 43)
unicode_179714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 4), 'unicode', u'dark cream')
unicode_179715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 18), 'unicode', u'#fff39a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179714, unicode_179715))
# Adding element type (key, value) (line 43)
unicode_179716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 4), 'unicode', u'indian red')
unicode_179717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 18), 'unicode', u'#850e04')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179716, unicode_179717))
# Adding element type (key, value) (line 43)
unicode_179718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 4), 'unicode', u'light lavendar')
unicode_179719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 22), 'unicode', u'#efc0fe')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179718, unicode_179719))
# Adding element type (key, value) (line 43)
unicode_179720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 4), 'unicode', u'poison green')
unicode_179721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 20), 'unicode', u'#40fd14')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179720, unicode_179721))
# Adding element type (key, value) (line 43)
unicode_179722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 4), 'unicode', u'baby puke green')
unicode_179723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 23), 'unicode', u'#b6c406')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179722, unicode_179723))
# Adding element type (key, value) (line 43)
unicode_179724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 4), 'unicode', u'bright yellow green')
unicode_179725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 27), 'unicode', u'#9dff00')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179724, unicode_179725))
# Adding element type (key, value) (line 43)
unicode_179726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 4), 'unicode', u'charcoal grey')
unicode_179727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 21), 'unicode', u'#3c4142')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179726, unicode_179727))
# Adding element type (key, value) (line 43)
unicode_179728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 4), 'unicode', u'squash')
unicode_179729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 14), 'unicode', u'#f2ab15')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179728, unicode_179729))
# Adding element type (key, value) (line 43)
unicode_179730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 4), 'unicode', u'cinnamon')
unicode_179731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 16), 'unicode', u'#ac4f06')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179730, unicode_179731))
# Adding element type (key, value) (line 43)
unicode_179732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 4), 'unicode', u'light pea green')
unicode_179733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 23), 'unicode', u'#c4fe82')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179732, unicode_179733))
# Adding element type (key, value) (line 43)
unicode_179734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 4), 'unicode', u'radioactive green')
unicode_179735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 25), 'unicode', u'#2cfa1f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179734, unicode_179735))
# Adding element type (key, value) (line 43)
unicode_179736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 4), 'unicode', u'raw sienna')
unicode_179737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 18), 'unicode', u'#9a6200')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179736, unicode_179737))
# Adding element type (key, value) (line 43)
unicode_179738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 4), 'unicode', u'baby purple')
unicode_179739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 19), 'unicode', u'#ca9bf7')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179738, unicode_179739))
# Adding element type (key, value) (line 43)
unicode_179740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 4), 'unicode', u'cocoa')
unicode_179741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 13), 'unicode', u'#875f42')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179740, unicode_179741))
# Adding element type (key, value) (line 43)
unicode_179742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 4), 'unicode', u'light royal blue')
unicode_179743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 24), 'unicode', u'#3a2efe')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179742, unicode_179743))
# Adding element type (key, value) (line 43)
unicode_179744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 4), 'unicode', u'orangeish')
unicode_179745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 17), 'unicode', u'#fd8d49')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179744, unicode_179745))
# Adding element type (key, value) (line 43)
unicode_179746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 4), 'unicode', u'rust brown')
unicode_179747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 18), 'unicode', u'#8b3103')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179746, unicode_179747))
# Adding element type (key, value) (line 43)
unicode_179748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 4), 'unicode', u'sand brown')
unicode_179749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 18), 'unicode', u'#cba560')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179748, unicode_179749))
# Adding element type (key, value) (line 43)
unicode_179750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 4), 'unicode', u'swamp')
unicode_179751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 13), 'unicode', u'#698339')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179750, unicode_179751))
# Adding element type (key, value) (line 43)
unicode_179752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 4), 'unicode', u'tealish green')
unicode_179753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 21), 'unicode', u'#0cdc73')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179752, unicode_179753))
# Adding element type (key, value) (line 43)
unicode_179754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 4), 'unicode', u'burnt siena')
unicode_179755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 19), 'unicode', u'#b75203')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179754, unicode_179755))
# Adding element type (key, value) (line 43)
unicode_179756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 4), 'unicode', u'camo')
unicode_179757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 12), 'unicode', u'#7f8f4e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179756, unicode_179757))
# Adding element type (key, value) (line 43)
unicode_179758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 4), 'unicode', u'dusk blue')
unicode_179759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 17), 'unicode', u'#26538d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179758, unicode_179759))
# Adding element type (key, value) (line 43)
unicode_179760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 4), 'unicode', u'fern')
unicode_179761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 12), 'unicode', u'#63a950')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179760, unicode_179761))
# Adding element type (key, value) (line 43)
unicode_179762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 4), 'unicode', u'old rose')
unicode_179763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 16), 'unicode', u'#c87f89')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179762, unicode_179763))
# Adding element type (key, value) (line 43)
unicode_179764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 4), 'unicode', u'pale light green')
unicode_179765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 24), 'unicode', u'#b1fc99')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179764, unicode_179765))
# Adding element type (key, value) (line 43)
unicode_179766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 4), 'unicode', u'peachy pink')
unicode_179767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 19), 'unicode', u'#ff9a8a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179766, unicode_179767))
# Adding element type (key, value) (line 43)
unicode_179768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 4), 'unicode', u'rosy pink')
unicode_179769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 17), 'unicode', u'#f6688e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179768, unicode_179769))
# Adding element type (key, value) (line 43)
unicode_179770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 4), 'unicode', u'light bluish green')
unicode_179771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 26), 'unicode', u'#76fda8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179770, unicode_179771))
# Adding element type (key, value) (line 43)
unicode_179772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 4), 'unicode', u'light bright green')
unicode_179773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 26), 'unicode', u'#53fe5c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179772, unicode_179773))
# Adding element type (key, value) (line 43)
unicode_179774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 4), 'unicode', u'light neon green')
unicode_179775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 24), 'unicode', u'#4efd54')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179774, unicode_179775))
# Adding element type (key, value) (line 43)
unicode_179776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 4), 'unicode', u'light seafoam')
unicode_179777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 21), 'unicode', u'#a0febf')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179776, unicode_179777))
# Adding element type (key, value) (line 43)
unicode_179778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 4), 'unicode', u'tiffany blue')
unicode_179779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 20), 'unicode', u'#7bf2da')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179778, unicode_179779))
# Adding element type (key, value) (line 43)
unicode_179780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 4), 'unicode', u'washed out green')
unicode_179781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 24), 'unicode', u'#bcf5a6')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179780, unicode_179781))
# Adding element type (key, value) (line 43)
unicode_179782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 4), 'unicode', u'browny orange')
unicode_179783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 21), 'unicode', u'#ca6b02')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179782, unicode_179783))
# Adding element type (key, value) (line 43)
unicode_179784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 4), 'unicode', u'nice blue')
unicode_179785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 17), 'unicode', u'#107ab0')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179784, unicode_179785))
# Adding element type (key, value) (line 43)
unicode_179786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 4), 'unicode', u'sapphire')
unicode_179787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 16), 'unicode', u'#2138ab')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179786, unicode_179787))
# Adding element type (key, value) (line 43)
unicode_179788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 4), 'unicode', u'greyish teal')
unicode_179789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 20), 'unicode', u'#719f91')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179788, unicode_179789))
# Adding element type (key, value) (line 43)
unicode_179790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 4), 'unicode', u'orangey yellow')
unicode_179791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 22), 'unicode', u'#fdb915')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179790, unicode_179791))
# Adding element type (key, value) (line 43)
unicode_179792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 4), 'unicode', u'parchment')
unicode_179793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 17), 'unicode', u'#fefcaf')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179792, unicode_179793))
# Adding element type (key, value) (line 43)
unicode_179794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 4), 'unicode', u'straw')
unicode_179795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 13), 'unicode', u'#fcf679')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179794, unicode_179795))
# Adding element type (key, value) (line 43)
unicode_179796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 4), 'unicode', u'very dark brown')
unicode_179797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 23), 'unicode', u'#1d0200')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179796, unicode_179797))
# Adding element type (key, value) (line 43)
unicode_179798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 4), 'unicode', u'terracota')
unicode_179799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 17), 'unicode', u'#cb6843')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179798, unicode_179799))
# Adding element type (key, value) (line 43)
unicode_179800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 4), 'unicode', u'ugly blue')
unicode_179801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 17), 'unicode', u'#31668a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179800, unicode_179801))
# Adding element type (key, value) (line 43)
unicode_179802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 4), 'unicode', u'clear blue')
unicode_179803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 18), 'unicode', u'#247afd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179802, unicode_179803))
# Adding element type (key, value) (line 43)
unicode_179804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 4), 'unicode', u'creme')
unicode_179805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 13), 'unicode', u'#ffffb6')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179804, unicode_179805))
# Adding element type (key, value) (line 43)
unicode_179806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 4), 'unicode', u'foam green')
unicode_179807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 18), 'unicode', u'#90fda9')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179806, unicode_179807))
# Adding element type (key, value) (line 43)
unicode_179808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 4), 'unicode', u'grey/green')
unicode_179809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 18), 'unicode', u'#86a17d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179808, unicode_179809))
# Adding element type (key, value) (line 43)
unicode_179810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 4), 'unicode', u'light gold')
unicode_179811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 18), 'unicode', u'#fddc5c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179810, unicode_179811))
# Adding element type (key, value) (line 43)
unicode_179812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 4), 'unicode', u'seafoam blue')
unicode_179813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 20), 'unicode', u'#78d1b6')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179812, unicode_179813))
# Adding element type (key, value) (line 43)
unicode_179814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 4), 'unicode', u'topaz')
unicode_179815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 13), 'unicode', u'#13bbaf')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179814, unicode_179815))
# Adding element type (key, value) (line 43)
unicode_179816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 4), 'unicode', u'violet pink')
unicode_179817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 19), 'unicode', u'#fb5ffc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179816, unicode_179817))
# Adding element type (key, value) (line 43)
unicode_179818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 4), 'unicode', u'wintergreen')
unicode_179819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 19), 'unicode', u'#20f986')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179818, unicode_179819))
# Adding element type (key, value) (line 43)
unicode_179820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 4), 'unicode', u'yellow tan')
unicode_179821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 18), 'unicode', u'#ffe36e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179820, unicode_179821))
# Adding element type (key, value) (line 43)
unicode_179822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 4), 'unicode', u'dark fuchsia')
unicode_179823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 20), 'unicode', u'#9d0759')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179822, unicode_179823))
# Adding element type (key, value) (line 43)
unicode_179824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 4), 'unicode', u'indigo blue')
unicode_179825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 19), 'unicode', u'#3a18b1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179824, unicode_179825))
# Adding element type (key, value) (line 43)
unicode_179826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 4), 'unicode', u'light yellowish green')
unicode_179827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 29), 'unicode', u'#c2ff89')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179826, unicode_179827))
# Adding element type (key, value) (line 43)
unicode_179828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 4), 'unicode', u'pale magenta')
unicode_179829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 20), 'unicode', u'#d767ad')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179828, unicode_179829))
# Adding element type (key, value) (line 43)
unicode_179830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 4), 'unicode', u'rich purple')
unicode_179831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 19), 'unicode', u'#720058')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179830, unicode_179831))
# Adding element type (key, value) (line 43)
unicode_179832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 4), 'unicode', u'sunflower yellow')
unicode_179833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 24), 'unicode', u'#ffda03')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179832, unicode_179833))
# Adding element type (key, value) (line 43)
unicode_179834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 4), 'unicode', u'green/blue')
unicode_179835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 18), 'unicode', u'#01c08d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179834, unicode_179835))
# Adding element type (key, value) (line 43)
unicode_179836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 4), 'unicode', u'leather')
unicode_179837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 15), 'unicode', u'#ac7434')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179836, unicode_179837))
# Adding element type (key, value) (line 43)
unicode_179838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 4), 'unicode', u'racing green')
unicode_179839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 20), 'unicode', u'#014600')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179838, unicode_179839))
# Adding element type (key, value) (line 43)
unicode_179840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 4), 'unicode', u'vivid purple')
unicode_179841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 20), 'unicode', u'#9900fa')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179840, unicode_179841))
# Adding element type (key, value) (line 43)
unicode_179842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 4), 'unicode', u'dark royal blue')
unicode_179843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 23), 'unicode', u'#02066f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179842, unicode_179843))
# Adding element type (key, value) (line 43)
unicode_179844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 4), 'unicode', u'hazel')
unicode_179845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 13), 'unicode', u'#8e7618')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179844, unicode_179845))
# Adding element type (key, value) (line 43)
unicode_179846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 4), 'unicode', u'muted pink')
unicode_179847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 18), 'unicode', u'#d1768f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179846, unicode_179847))
# Adding element type (key, value) (line 43)
unicode_179848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 4), 'unicode', u'booger green')
unicode_179849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 20), 'unicode', u'#96b403')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179848, unicode_179849))
# Adding element type (key, value) (line 43)
unicode_179850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 4), 'unicode', u'canary')
unicode_179851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 14), 'unicode', u'#fdff63')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179850, unicode_179851))
# Adding element type (key, value) (line 43)
unicode_179852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 4), 'unicode', u'cool grey')
unicode_179853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 17), 'unicode', u'#95a3a6')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179852, unicode_179853))
# Adding element type (key, value) (line 43)
unicode_179854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 4), 'unicode', u'dark taupe')
unicode_179855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 18), 'unicode', u'#7f684e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179854, unicode_179855))
# Adding element type (key, value) (line 43)
unicode_179856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 4), 'unicode', u'darkish purple')
unicode_179857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 22), 'unicode', u'#751973')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179856, unicode_179857))
# Adding element type (key, value) (line 43)
unicode_179858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 4), 'unicode', u'true green')
unicode_179859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 18), 'unicode', u'#089404')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179858, unicode_179859))
# Adding element type (key, value) (line 43)
unicode_179860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 4), 'unicode', u'coral pink')
unicode_179861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 18), 'unicode', u'#ff6163')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179860, unicode_179861))
# Adding element type (key, value) (line 43)
unicode_179862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 4), 'unicode', u'dark sage')
unicode_179863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 17), 'unicode', u'#598556')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179862, unicode_179863))
# Adding element type (key, value) (line 43)
unicode_179864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 4), 'unicode', u'dark slate blue')
unicode_179865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 23), 'unicode', u'#214761')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179864, unicode_179865))
# Adding element type (key, value) (line 43)
unicode_179866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 4), 'unicode', u'flat blue')
unicode_179867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 17), 'unicode', u'#3c73a8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179866, unicode_179867))
# Adding element type (key, value) (line 43)
unicode_179868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 4), 'unicode', u'mushroom')
unicode_179869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 16), 'unicode', u'#ba9e88')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179868, unicode_179869))
# Adding element type (key, value) (line 43)
unicode_179870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 4), 'unicode', u'rich blue')
unicode_179871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 17), 'unicode', u'#021bf9')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179870, unicode_179871))
# Adding element type (key, value) (line 43)
unicode_179872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 4), 'unicode', u'dirty purple')
unicode_179873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 20), 'unicode', u'#734a65')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179872, unicode_179873))
# Adding element type (key, value) (line 43)
unicode_179874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 4), 'unicode', u'greenblue')
unicode_179875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 17), 'unicode', u'#23c48b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179874, unicode_179875))
# Adding element type (key, value) (line 43)
unicode_179876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 4), 'unicode', u'icky green')
unicode_179877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 18), 'unicode', u'#8fae22')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179876, unicode_179877))
# Adding element type (key, value) (line 43)
unicode_179878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 4), 'unicode', u'light khaki')
unicode_179879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 19), 'unicode', u'#e6f2a2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179878, unicode_179879))
# Adding element type (key, value) (line 43)
unicode_179880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 4), 'unicode', u'warm blue')
unicode_179881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 17), 'unicode', u'#4b57db')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179880, unicode_179881))
# Adding element type (key, value) (line 43)
unicode_179882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 4), 'unicode', u'dark hot pink')
unicode_179883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 21), 'unicode', u'#d90166')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179882, unicode_179883))
# Adding element type (key, value) (line 43)
unicode_179884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 4), 'unicode', u'deep sea blue')
unicode_179885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 21), 'unicode', u'#015482')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179884, unicode_179885))
# Adding element type (key, value) (line 43)
unicode_179886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 4), 'unicode', u'carmine')
unicode_179887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 15), 'unicode', u'#9d0216')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179886, unicode_179887))
# Adding element type (key, value) (line 43)
unicode_179888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 4), 'unicode', u'dark yellow green')
unicode_179889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 25), 'unicode', u'#728f02')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179888, unicode_179889))
# Adding element type (key, value) (line 43)
unicode_179890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 4), 'unicode', u'pale peach')
unicode_179891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 18), 'unicode', u'#ffe5ad')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179890, unicode_179891))
# Adding element type (key, value) (line 43)
unicode_179892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 4), 'unicode', u'plum purple')
unicode_179893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 19), 'unicode', u'#4e0550')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179892, unicode_179893))
# Adding element type (key, value) (line 43)
unicode_179894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 4), 'unicode', u'golden rod')
unicode_179895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 18), 'unicode', u'#f9bc08')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179894, unicode_179895))
# Adding element type (key, value) (line 43)
unicode_179896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 4), 'unicode', u'neon red')
unicode_179897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 16), 'unicode', u'#ff073a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179896, unicode_179897))
# Adding element type (key, value) (line 43)
unicode_179898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 4), 'unicode', u'old pink')
unicode_179899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 16), 'unicode', u'#c77986')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179898, unicode_179899))
# Adding element type (key, value) (line 43)
unicode_179900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 4), 'unicode', u'very pale blue')
unicode_179901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 22), 'unicode', u'#d6fffe')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179900, unicode_179901))
# Adding element type (key, value) (line 43)
unicode_179902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 4), 'unicode', u'blood orange')
unicode_179903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 20), 'unicode', u'#fe4b03')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179902, unicode_179903))
# Adding element type (key, value) (line 43)
unicode_179904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 4), 'unicode', u'grapefruit')
unicode_179905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 18), 'unicode', u'#fd5956')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179904, unicode_179905))
# Adding element type (key, value) (line 43)
unicode_179906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 4), 'unicode', u'sand yellow')
unicode_179907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 19), 'unicode', u'#fce166')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179906, unicode_179907))
# Adding element type (key, value) (line 43)
unicode_179908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 4), 'unicode', u'clay brown')
unicode_179909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 18), 'unicode', u'#b2713d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179908, unicode_179909))
# Adding element type (key, value) (line 43)
unicode_179910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 4), 'unicode', u'dark blue grey')
unicode_179911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 22), 'unicode', u'#1f3b4d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179910, unicode_179911))
# Adding element type (key, value) (line 43)
unicode_179912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 4), 'unicode', u'flat green')
unicode_179913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 18), 'unicode', u'#699d4c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179912, unicode_179913))
# Adding element type (key, value) (line 43)
unicode_179914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 4), 'unicode', u'light green blue')
unicode_179915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 24), 'unicode', u'#56fca2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179914, unicode_179915))
# Adding element type (key, value) (line 43)
unicode_179916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 4), 'unicode', u'warm pink')
unicode_179917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 17), 'unicode', u'#fb5581')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179916, unicode_179917))
# Adding element type (key, value) (line 43)
unicode_179918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 4), 'unicode', u'dodger blue')
unicode_179919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 19), 'unicode', u'#3e82fc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179918, unicode_179919))
# Adding element type (key, value) (line 43)
unicode_179920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 4), 'unicode', u'gross green')
unicode_179921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 19), 'unicode', u'#a0bf16')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179920, unicode_179921))
# Adding element type (key, value) (line 43)
unicode_179922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 4), 'unicode', u'ice')
unicode_179923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 11), 'unicode', u'#d6fffa')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179922, unicode_179923))
# Adding element type (key, value) (line 43)
unicode_179924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 4), 'unicode', u'metallic blue')
unicode_179925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 21), 'unicode', u'#4f738e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179924, unicode_179925))
# Adding element type (key, value) (line 43)
unicode_179926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 4), 'unicode', u'pale salmon')
unicode_179927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 19), 'unicode', u'#ffb19a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179926, unicode_179927))
# Adding element type (key, value) (line 43)
unicode_179928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 4), 'unicode', u'sap green')
unicode_179929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 17), 'unicode', u'#5c8b15')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179928, unicode_179929))
# Adding element type (key, value) (line 43)
unicode_179930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 4), 'unicode', u'algae')
unicode_179931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 13), 'unicode', u'#54ac68')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179930, unicode_179931))
# Adding element type (key, value) (line 43)
unicode_179932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 4), 'unicode', u'bluey grey')
unicode_179933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 18), 'unicode', u'#89a0b0')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179932, unicode_179933))
# Adding element type (key, value) (line 43)
unicode_179934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 4), 'unicode', u'greeny grey')
unicode_179935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 19), 'unicode', u'#7ea07a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179934, unicode_179935))
# Adding element type (key, value) (line 43)
unicode_179936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 4), 'unicode', u'highlighter green')
unicode_179937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 25), 'unicode', u'#1bfc06')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179936, unicode_179937))
# Adding element type (key, value) (line 43)
unicode_179938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 4), 'unicode', u'light light blue')
unicode_179939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 24), 'unicode', u'#cafffb')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179938, unicode_179939))
# Adding element type (key, value) (line 43)
unicode_179940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 4), 'unicode', u'light mint')
unicode_179941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 18), 'unicode', u'#b6ffbb')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179940, unicode_179941))
# Adding element type (key, value) (line 43)
unicode_179942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 4), 'unicode', u'raw umber')
unicode_179943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 17), 'unicode', u'#a75e09')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179942, unicode_179943))
# Adding element type (key, value) (line 43)
unicode_179944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 4), 'unicode', u'vivid blue')
unicode_179945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 18), 'unicode', u'#152eff')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179944, unicode_179945))
# Adding element type (key, value) (line 43)
unicode_179946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 4), 'unicode', u'deep lavender')
unicode_179947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 21), 'unicode', u'#8d5eb7')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179946, unicode_179947))
# Adding element type (key, value) (line 43)
unicode_179948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 4), 'unicode', u'dull teal')
unicode_179949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 17), 'unicode', u'#5f9e8f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179948, unicode_179949))
# Adding element type (key, value) (line 43)
unicode_179950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 4), 'unicode', u'light greenish blue')
unicode_179951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 27), 'unicode', u'#63f7b4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179950, unicode_179951))
# Adding element type (key, value) (line 43)
unicode_179952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 4), 'unicode', u'mud green')
unicode_179953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 17), 'unicode', u'#606602')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179952, unicode_179953))
# Adding element type (key, value) (line 43)
unicode_179954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 4), 'unicode', u'pinky')
unicode_179955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 13), 'unicode', u'#fc86aa')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179954, unicode_179955))
# Adding element type (key, value) (line 43)
unicode_179956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 4), 'unicode', u'red wine')
unicode_179957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 16), 'unicode', u'#8c0034')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179956, unicode_179957))
# Adding element type (key, value) (line 43)
unicode_179958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 4), 'unicode', u'shit green')
unicode_179959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 18), 'unicode', u'#758000')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179958, unicode_179959))
# Adding element type (key, value) (line 43)
unicode_179960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 4), 'unicode', u'tan brown')
unicode_179961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 17), 'unicode', u'#ab7e4c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179960, unicode_179961))
# Adding element type (key, value) (line 43)
unicode_179962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 4), 'unicode', u'darkblue')
unicode_179963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 16), 'unicode', u'#030764')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179962, unicode_179963))
# Adding element type (key, value) (line 43)
unicode_179964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 4), 'unicode', u'rosa')
unicode_179965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 12), 'unicode', u'#fe86a4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179964, unicode_179965))
# Adding element type (key, value) (line 43)
unicode_179966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 4), 'unicode', u'lipstick')
unicode_179967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 16), 'unicode', u'#d5174e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179966, unicode_179967))
# Adding element type (key, value) (line 43)
unicode_179968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 4), 'unicode', u'pale mauve')
unicode_179969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 18), 'unicode', u'#fed0fc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179968, unicode_179969))
# Adding element type (key, value) (line 43)
unicode_179970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 4), 'unicode', u'claret')
unicode_179971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 14), 'unicode', u'#680018')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179970, unicode_179971))
# Adding element type (key, value) (line 43)
unicode_179972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 4), 'unicode', u'dandelion')
unicode_179973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 17), 'unicode', u'#fedf08')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179972, unicode_179973))
# Adding element type (key, value) (line 43)
unicode_179974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 4), 'unicode', u'orangered')
unicode_179975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 17), 'unicode', u'#fe420f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179974, unicode_179975))
# Adding element type (key, value) (line 43)
unicode_179976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 4), 'unicode', u'poop green')
unicode_179977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 18), 'unicode', u'#6f7c00')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179976, unicode_179977))
# Adding element type (key, value) (line 43)
unicode_179978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 4), 'unicode', u'ruby')
unicode_179979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 12), 'unicode', u'#ca0147')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179978, unicode_179979))
# Adding element type (key, value) (line 43)
unicode_179980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 4), 'unicode', u'dark')
unicode_179981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 12), 'unicode', u'#1b2431')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179980, unicode_179981))
# Adding element type (key, value) (line 43)
unicode_179982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 4), 'unicode', u'greenish turquoise')
unicode_179983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 26), 'unicode', u'#00fbb0')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179982, unicode_179983))
# Adding element type (key, value) (line 43)
unicode_179984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 4), 'unicode', u'pastel red')
unicode_179985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 18), 'unicode', u'#db5856')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179984, unicode_179985))
# Adding element type (key, value) (line 43)
unicode_179986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 4), 'unicode', u'piss yellow')
unicode_179987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 19), 'unicode', u'#ddd618')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179986, unicode_179987))
# Adding element type (key, value) (line 43)
unicode_179988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 4), 'unicode', u'bright cyan')
unicode_179989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 19), 'unicode', u'#41fdfe')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179988, unicode_179989))
# Adding element type (key, value) (line 43)
unicode_179990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 4), 'unicode', u'dark coral')
unicode_179991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 18), 'unicode', u'#cf524e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179990, unicode_179991))
# Adding element type (key, value) (line 43)
unicode_179992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 4), 'unicode', u'algae green')
unicode_179993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 19), 'unicode', u'#21c36f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179992, unicode_179993))
# Adding element type (key, value) (line 43)
unicode_179994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 4), 'unicode', u'darkish red')
unicode_179995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 19), 'unicode', u'#a90308')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179994, unicode_179995))
# Adding element type (key, value) (line 43)
unicode_179996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 4), 'unicode', u'reddy brown')
unicode_179997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 19), 'unicode', u'#6e1005')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179996, unicode_179997))
# Adding element type (key, value) (line 43)
unicode_179998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 4), 'unicode', u'blush pink')
unicode_179999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 18), 'unicode', u'#fe828c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_179998, unicode_179999))
# Adding element type (key, value) (line 43)
unicode_180000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 4), 'unicode', u'camouflage green')
unicode_180001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 24), 'unicode', u'#4b6113')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180000, unicode_180001))
# Adding element type (key, value) (line 43)
unicode_180002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 4), 'unicode', u'lawn green')
unicode_180003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 18), 'unicode', u'#4da409')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180002, unicode_180003))
# Adding element type (key, value) (line 43)
unicode_180004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 4), 'unicode', u'putty')
unicode_180005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 13), 'unicode', u'#beae8a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180004, unicode_180005))
# Adding element type (key, value) (line 43)
unicode_180006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 4), 'unicode', u'vibrant blue')
unicode_180007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 20), 'unicode', u'#0339f8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180006, unicode_180007))
# Adding element type (key, value) (line 43)
unicode_180008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 4), 'unicode', u'dark sand')
unicode_180009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 17), 'unicode', u'#a88f59')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180008, unicode_180009))
# Adding element type (key, value) (line 43)
unicode_180010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 4), 'unicode', u'purple/blue')
unicode_180011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 19), 'unicode', u'#5d21d0')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180010, unicode_180011))
# Adding element type (key, value) (line 43)
unicode_180012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 4), 'unicode', u'saffron')
unicode_180013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 15), 'unicode', u'#feb209')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180012, unicode_180013))
# Adding element type (key, value) (line 43)
unicode_180014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 4), 'unicode', u'twilight')
unicode_180015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 16), 'unicode', u'#4e518b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180014, unicode_180015))
# Adding element type (key, value) (line 43)
unicode_180016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 4), 'unicode', u'warm brown')
unicode_180017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 18), 'unicode', u'#964e02')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180016, unicode_180017))
# Adding element type (key, value) (line 43)
unicode_180018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 4), 'unicode', u'bluegrey')
unicode_180019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 16), 'unicode', u'#85a3b2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180018, unicode_180019))
# Adding element type (key, value) (line 43)
unicode_180020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 4), 'unicode', u'bubble gum pink')
unicode_180021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 23), 'unicode', u'#ff69af')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180020, unicode_180021))
# Adding element type (key, value) (line 43)
unicode_180022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 4), 'unicode', u'duck egg blue')
unicode_180023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 21), 'unicode', u'#c3fbf4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180022, unicode_180023))
# Adding element type (key, value) (line 43)
unicode_180024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 4), 'unicode', u'greenish cyan')
unicode_180025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 21), 'unicode', u'#2afeb7')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180024, unicode_180025))
# Adding element type (key, value) (line 43)
unicode_180026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 4), 'unicode', u'petrol')
unicode_180027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 14), 'unicode', u'#005f6a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180026, unicode_180027))
# Adding element type (key, value) (line 43)
unicode_180028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 4), 'unicode', u'royal')
unicode_180029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 13), 'unicode', u'#0c1793')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180028, unicode_180029))
# Adding element type (key, value) (line 43)
unicode_180030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 4), 'unicode', u'butter')
unicode_180031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 14), 'unicode', u'#ffff81')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180030, unicode_180031))
# Adding element type (key, value) (line 43)
unicode_180032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 4), 'unicode', u'dusty orange')
unicode_180033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 20), 'unicode', u'#f0833a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180032, unicode_180033))
# Adding element type (key, value) (line 43)
unicode_180034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 4), 'unicode', u'off yellow')
unicode_180035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 18), 'unicode', u'#f1f33f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180034, unicode_180035))
# Adding element type (key, value) (line 43)
unicode_180036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 4), 'unicode', u'pale olive green')
unicode_180037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 24), 'unicode', u'#b1d27b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180036, unicode_180037))
# Adding element type (key, value) (line 43)
unicode_180038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 4), 'unicode', u'orangish')
unicode_180039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 16), 'unicode', u'#fc824a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180038, unicode_180039))
# Adding element type (key, value) (line 43)
unicode_180040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 4), 'unicode', u'leaf')
unicode_180041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 12), 'unicode', u'#71aa34')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180040, unicode_180041))
# Adding element type (key, value) (line 43)
unicode_180042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 4), 'unicode', u'light blue grey')
unicode_180043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 23), 'unicode', u'#b7c9e2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180042, unicode_180043))
# Adding element type (key, value) (line 43)
unicode_180044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 4), 'unicode', u'dried blood')
unicode_180045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 19), 'unicode', u'#4b0101')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180044, unicode_180045))
# Adding element type (key, value) (line 43)
unicode_180046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 4), 'unicode', u'lightish purple')
unicode_180047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 23), 'unicode', u'#a552e6')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180046, unicode_180047))
# Adding element type (key, value) (line 43)
unicode_180048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 4), 'unicode', u'rusty red')
unicode_180049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 17), 'unicode', u'#af2f0d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180048, unicode_180049))
# Adding element type (key, value) (line 43)
unicode_180050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 4), 'unicode', u'lavender blue')
unicode_180051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 21), 'unicode', u'#8b88f8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180050, unicode_180051))
# Adding element type (key, value) (line 43)
unicode_180052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 4), 'unicode', u'light grass green')
unicode_180053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 25), 'unicode', u'#9af764')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180052, unicode_180053))
# Adding element type (key, value) (line 43)
unicode_180054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 4), 'unicode', u'light mint green')
unicode_180055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 24), 'unicode', u'#a6fbb2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180054, unicode_180055))
# Adding element type (key, value) (line 43)
unicode_180056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 4), 'unicode', u'sunflower')
unicode_180057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 17), 'unicode', u'#ffc512')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180056, unicode_180057))
# Adding element type (key, value) (line 43)
unicode_180058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 4), 'unicode', u'velvet')
unicode_180059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 14), 'unicode', u'#750851')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180058, unicode_180059))
# Adding element type (key, value) (line 43)
unicode_180060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 4), 'unicode', u'brick orange')
unicode_180061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 20), 'unicode', u'#c14a09')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180060, unicode_180061))
# Adding element type (key, value) (line 43)
unicode_180062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 4), 'unicode', u'lightish red')
unicode_180063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 20), 'unicode', u'#fe2f4a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180062, unicode_180063))
# Adding element type (key, value) (line 43)
unicode_180064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 4), 'unicode', u'pure blue')
unicode_180065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 17), 'unicode', u'#0203e2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180064, unicode_180065))
# Adding element type (key, value) (line 43)
unicode_180066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 4), 'unicode', u'twilight blue')
unicode_180067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 21), 'unicode', u'#0a437a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180066, unicode_180067))
# Adding element type (key, value) (line 43)
unicode_180068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 4), 'unicode', u'violet red')
unicode_180069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 18), 'unicode', u'#a50055')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180068, unicode_180069))
# Adding element type (key, value) (line 43)
unicode_180070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 4), 'unicode', u'yellowy brown')
unicode_180071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 21), 'unicode', u'#ae8b0c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180070, unicode_180071))
# Adding element type (key, value) (line 43)
unicode_180072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 4), 'unicode', u'carnation')
unicode_180073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 17), 'unicode', u'#fd798f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180072, unicode_180073))
# Adding element type (key, value) (line 43)
unicode_180074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 4), 'unicode', u'muddy yellow')
unicode_180075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 20), 'unicode', u'#bfac05')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180074, unicode_180075))
# Adding element type (key, value) (line 43)
unicode_180076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 4), 'unicode', u'dark seafoam green')
unicode_180077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 26), 'unicode', u'#3eaf76')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180076, unicode_180077))
# Adding element type (key, value) (line 43)
unicode_180078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 4), 'unicode', u'deep rose')
unicode_180079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 17), 'unicode', u'#c74767')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180078, unicode_180079))
# Adding element type (key, value) (line 43)
unicode_180080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 4), 'unicode', u'dusty red')
unicode_180081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 17), 'unicode', u'#b9484e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180080, unicode_180081))
# Adding element type (key, value) (line 43)
unicode_180082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 4), 'unicode', u'grey/blue')
unicode_180083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 17), 'unicode', u'#647d8e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180082, unicode_180083))
# Adding element type (key, value) (line 43)
unicode_180084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 4), 'unicode', u'lemon lime')
unicode_180085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 18), 'unicode', u'#bffe28')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180084, unicode_180085))
# Adding element type (key, value) (line 43)
unicode_180086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 4), 'unicode', u'purple/pink')
unicode_180087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 19), 'unicode', u'#d725de')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180086, unicode_180087))
# Adding element type (key, value) (line 43)
unicode_180088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 4), 'unicode', u'brown yellow')
unicode_180089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 20), 'unicode', u'#b29705')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180088, unicode_180089))
# Adding element type (key, value) (line 43)
unicode_180090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 4), 'unicode', u'purple brown')
unicode_180091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 20), 'unicode', u'#673a3f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180090, unicode_180091))
# Adding element type (key, value) (line 43)
unicode_180092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 4), 'unicode', u'wisteria')
unicode_180093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 16), 'unicode', u'#a87dc2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180092, unicode_180093))
# Adding element type (key, value) (line 43)
unicode_180094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 4), 'unicode', u'banana yellow')
unicode_180095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 21), 'unicode', u'#fafe4b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180094, unicode_180095))
# Adding element type (key, value) (line 43)
unicode_180096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 4), 'unicode', u'lipstick red')
unicode_180097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 20), 'unicode', u'#c0022f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180096, unicode_180097))
# Adding element type (key, value) (line 43)
unicode_180098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 4), 'unicode', u'water blue')
unicode_180099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 18), 'unicode', u'#0e87cc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180098, unicode_180099))
# Adding element type (key, value) (line 43)
unicode_180100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 4), 'unicode', u'brown grey')
unicode_180101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 18), 'unicode', u'#8d8468')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180100, unicode_180101))
# Adding element type (key, value) (line 43)
unicode_180102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 4), 'unicode', u'vibrant purple')
unicode_180103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 22), 'unicode', u'#ad03de')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180102, unicode_180103))
# Adding element type (key, value) (line 43)
unicode_180104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 4), 'unicode', u'baby green')
unicode_180105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 18), 'unicode', u'#8cff9e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180104, unicode_180105))
# Adding element type (key, value) (line 43)
unicode_180106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 4), 'unicode', u'barf green')
unicode_180107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 18), 'unicode', u'#94ac02')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180106, unicode_180107))
# Adding element type (key, value) (line 43)
unicode_180108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 4), 'unicode', u'eggshell blue')
unicode_180109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 21), 'unicode', u'#c4fff7')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180108, unicode_180109))
# Adding element type (key, value) (line 43)
unicode_180110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 4), 'unicode', u'sandy yellow')
unicode_180111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 20), 'unicode', u'#fdee73')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180110, unicode_180111))
# Adding element type (key, value) (line 43)
unicode_180112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 4), 'unicode', u'cool green')
unicode_180113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 18), 'unicode', u'#33b864')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180112, unicode_180113))
# Adding element type (key, value) (line 43)
unicode_180114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 4), 'unicode', u'pale')
unicode_180115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 12), 'unicode', u'#fff9d0')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180114, unicode_180115))
# Adding element type (key, value) (line 43)
unicode_180116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 4), 'unicode', u'blue/grey')
unicode_180117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 17), 'unicode', u'#758da3')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180116, unicode_180117))
# Adding element type (key, value) (line 43)
unicode_180118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 4), 'unicode', u'hot magenta')
unicode_180119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 19), 'unicode', u'#f504c9')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180118, unicode_180119))
# Adding element type (key, value) (line 43)
unicode_180120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 4), 'unicode', u'greyblue')
unicode_180121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 16), 'unicode', u'#77a1b5')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180120, unicode_180121))
# Adding element type (key, value) (line 43)
unicode_180122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 4), 'unicode', u'purpley')
unicode_180123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 15), 'unicode', u'#8756e4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180122, unicode_180123))
# Adding element type (key, value) (line 43)
unicode_180124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 4), 'unicode', u'baby shit green')
unicode_180125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 23), 'unicode', u'#889717')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180124, unicode_180125))
# Adding element type (key, value) (line 43)
unicode_180126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 4), 'unicode', u'brownish pink')
unicode_180127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 21), 'unicode', u'#c27e79')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180126, unicode_180127))
# Adding element type (key, value) (line 43)
unicode_180128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 4), 'unicode', u'dark aquamarine')
unicode_180129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 23), 'unicode', u'#017371')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180128, unicode_180129))
# Adding element type (key, value) (line 43)
unicode_180130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 4), 'unicode', u'diarrhea')
unicode_180131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 16), 'unicode', u'#9f8303')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180130, unicode_180131))
# Adding element type (key, value) (line 43)
unicode_180132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 4), 'unicode', u'light mustard')
unicode_180133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 21), 'unicode', u'#f7d560')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180132, unicode_180133))
# Adding element type (key, value) (line 43)
unicode_180134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 4), 'unicode', u'pale sky blue')
unicode_180135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 21), 'unicode', u'#bdf6fe')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180134, unicode_180135))
# Adding element type (key, value) (line 43)
unicode_180136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 4), 'unicode', u'turtle green')
unicode_180137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 20), 'unicode', u'#75b84f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180136, unicode_180137))
# Adding element type (key, value) (line 43)
unicode_180138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 4), 'unicode', u'bright olive')
unicode_180139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 20), 'unicode', u'#9cbb04')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180138, unicode_180139))
# Adding element type (key, value) (line 43)
unicode_180140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 4), 'unicode', u'dark grey blue')
unicode_180141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 22), 'unicode', u'#29465b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180140, unicode_180141))
# Adding element type (key, value) (line 43)
unicode_180142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 4), 'unicode', u'greeny brown')
unicode_180143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 20), 'unicode', u'#696006')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180142, unicode_180143))
# Adding element type (key, value) (line 43)
unicode_180144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 4), 'unicode', u'lemon green')
unicode_180145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 19), 'unicode', u'#adf802')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180144, unicode_180145))
# Adding element type (key, value) (line 43)
unicode_180146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 4), 'unicode', u'light periwinkle')
unicode_180147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 24), 'unicode', u'#c1c6fc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180146, unicode_180147))
# Adding element type (key, value) (line 43)
unicode_180148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 4), 'unicode', u'seaweed green')
unicode_180149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 21), 'unicode', u'#35ad6b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180148, unicode_180149))
# Adding element type (key, value) (line 43)
unicode_180150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 4), 'unicode', u'sunshine yellow')
unicode_180151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 23), 'unicode', u'#fffd37')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180150, unicode_180151))
# Adding element type (key, value) (line 43)
unicode_180152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 4), 'unicode', u'ugly purple')
unicode_180153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 19), 'unicode', u'#a442a0')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180152, unicode_180153))
# Adding element type (key, value) (line 43)
unicode_180154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 4), 'unicode', u'medium pink')
unicode_180155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 19), 'unicode', u'#f36196')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180154, unicode_180155))
# Adding element type (key, value) (line 43)
unicode_180156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 4), 'unicode', u'puke brown')
unicode_180157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 18), 'unicode', u'#947706')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180156, unicode_180157))
# Adding element type (key, value) (line 43)
unicode_180158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 4), 'unicode', u'very light pink')
unicode_180159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 23), 'unicode', u'#fff4f2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180158, unicode_180159))
# Adding element type (key, value) (line 43)
unicode_180160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 4), 'unicode', u'viridian')
unicode_180161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 16), 'unicode', u'#1e9167')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180160, unicode_180161))
# Adding element type (key, value) (line 43)
unicode_180162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 4), 'unicode', u'bile')
unicode_180163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 12), 'unicode', u'#b5c306')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180162, unicode_180163))
# Adding element type (key, value) (line 43)
unicode_180164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 4), 'unicode', u'faded yellow')
unicode_180165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 20), 'unicode', u'#feff7f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180164, unicode_180165))
# Adding element type (key, value) (line 43)
unicode_180166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 4), 'unicode', u'very pale green')
unicode_180167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 23), 'unicode', u'#cffdbc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180166, unicode_180167))
# Adding element type (key, value) (line 43)
unicode_180168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 4), 'unicode', u'vibrant green')
unicode_180169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 21), 'unicode', u'#0add08')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180168, unicode_180169))
# Adding element type (key, value) (line 43)
unicode_180170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 4), 'unicode', u'bright lime')
unicode_180171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 19), 'unicode', u'#87fd05')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180170, unicode_180171))
# Adding element type (key, value) (line 43)
unicode_180172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 4), 'unicode', u'spearmint')
unicode_180173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 17), 'unicode', u'#1ef876')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180172, unicode_180173))
# Adding element type (key, value) (line 43)
unicode_180174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 4), 'unicode', u'light aquamarine')
unicode_180175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 24), 'unicode', u'#7bfdc7')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180174, unicode_180175))
# Adding element type (key, value) (line 43)
unicode_180176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 4), 'unicode', u'light sage')
unicode_180177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 18), 'unicode', u'#bcecac')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180176, unicode_180177))
# Adding element type (key, value) (line 43)
unicode_180178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 4), 'unicode', u'yellowgreen')
unicode_180179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 19), 'unicode', u'#bbf90f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180178, unicode_180179))
# Adding element type (key, value) (line 43)
unicode_180180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 4), 'unicode', u'baby poo')
unicode_180181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 16), 'unicode', u'#ab9004')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180180, unicode_180181))
# Adding element type (key, value) (line 43)
unicode_180182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 4), 'unicode', u'dark seafoam')
unicode_180183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 20), 'unicode', u'#1fb57a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180182, unicode_180183))
# Adding element type (key, value) (line 43)
unicode_180184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 4), 'unicode', u'deep teal')
unicode_180185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 17), 'unicode', u'#00555a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180184, unicode_180185))
# Adding element type (key, value) (line 43)
unicode_180186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 4), 'unicode', u'heather')
unicode_180187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 15), 'unicode', u'#a484ac')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180186, unicode_180187))
# Adding element type (key, value) (line 43)
unicode_180188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 4), 'unicode', u'rust orange')
unicode_180189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 19), 'unicode', u'#c45508')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180188, unicode_180189))
# Adding element type (key, value) (line 43)
unicode_180190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 4), 'unicode', u'dirty blue')
unicode_180191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 18), 'unicode', u'#3f829d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180190, unicode_180191))
# Adding element type (key, value) (line 43)
unicode_180192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 4), 'unicode', u'fern green')
unicode_180193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 18), 'unicode', u'#548d44')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180192, unicode_180193))
# Adding element type (key, value) (line 43)
unicode_180194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 4), 'unicode', u'bright lilac')
unicode_180195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 20), 'unicode', u'#c95efb')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180194, unicode_180195))
# Adding element type (key, value) (line 43)
unicode_180196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 4), 'unicode', u'weird green')
unicode_180197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 19), 'unicode', u'#3ae57f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180196, unicode_180197))
# Adding element type (key, value) (line 43)
unicode_180198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 4), 'unicode', u'peacock blue')
unicode_180199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 20), 'unicode', u'#016795')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180198, unicode_180199))
# Adding element type (key, value) (line 43)
unicode_180200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 4), 'unicode', u'avocado green')
unicode_180201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 21), 'unicode', u'#87a922')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180200, unicode_180201))
# Adding element type (key, value) (line 43)
unicode_180202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 4), 'unicode', u'faded orange')
unicode_180203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 20), 'unicode', u'#f0944d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180202, unicode_180203))
# Adding element type (key, value) (line 43)
unicode_180204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 4), 'unicode', u'grape purple')
unicode_180205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 20), 'unicode', u'#5d1451')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180204, unicode_180205))
# Adding element type (key, value) (line 43)
unicode_180206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 4), 'unicode', u'hot green')
unicode_180207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 17), 'unicode', u'#25ff29')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180206, unicode_180207))
# Adding element type (key, value) (line 43)
unicode_180208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 4), 'unicode', u'lime yellow')
unicode_180209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 19), 'unicode', u'#d0fe1d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180208, unicode_180209))
# Adding element type (key, value) (line 43)
unicode_180210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 4), 'unicode', u'mango')
unicode_180211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 13), 'unicode', u'#ffa62b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180210, unicode_180211))
# Adding element type (key, value) (line 43)
unicode_180212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 4), 'unicode', u'shamrock')
unicode_180213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 16), 'unicode', u'#01b44c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180212, unicode_180213))
# Adding element type (key, value) (line 43)
unicode_180214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 4), 'unicode', u'bubblegum')
unicode_180215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 17), 'unicode', u'#ff6cb5')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180214, unicode_180215))
# Adding element type (key, value) (line 43)
unicode_180216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 4), 'unicode', u'purplish brown')
unicode_180217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 22), 'unicode', u'#6b4247')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180216, unicode_180217))
# Adding element type (key, value) (line 43)
unicode_180218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 4), 'unicode', u'vomit yellow')
unicode_180219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 20), 'unicode', u'#c7c10c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180218, unicode_180219))
# Adding element type (key, value) (line 43)
unicode_180220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 4), 'unicode', u'pale cyan')
unicode_180221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 17), 'unicode', u'#b7fffa')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180220, unicode_180221))
# Adding element type (key, value) (line 43)
unicode_180222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 4), 'unicode', u'key lime')
unicode_180223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 16), 'unicode', u'#aeff6e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180222, unicode_180223))
# Adding element type (key, value) (line 43)
unicode_180224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 4), 'unicode', u'tomato red')
unicode_180225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 18), 'unicode', u'#ec2d01')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180224, unicode_180225))
# Adding element type (key, value) (line 43)
unicode_180226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 4), 'unicode', u'lightgreen')
unicode_180227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 18), 'unicode', u'#76ff7b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180226, unicode_180227))
# Adding element type (key, value) (line 43)
unicode_180228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 4), 'unicode', u'merlot')
unicode_180229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 14), 'unicode', u'#730039')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180228, unicode_180229))
# Adding element type (key, value) (line 43)
unicode_180230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 4), 'unicode', u'night blue')
unicode_180231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 18), 'unicode', u'#040348')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180230, unicode_180231))
# Adding element type (key, value) (line 43)
unicode_180232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 4), 'unicode', u'purpleish pink')
unicode_180233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 22), 'unicode', u'#df4ec8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180232, unicode_180233))
# Adding element type (key, value) (line 43)
unicode_180234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 4), 'unicode', u'apple')
unicode_180235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 13), 'unicode', u'#6ecb3c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180234, unicode_180235))
# Adding element type (key, value) (line 43)
unicode_180236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 4), 'unicode', u'baby poop green')
unicode_180237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 23), 'unicode', u'#8f9805')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180236, unicode_180237))
# Adding element type (key, value) (line 43)
unicode_180238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 4), 'unicode', u'green apple')
unicode_180239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 19), 'unicode', u'#5edc1f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180238, unicode_180239))
# Adding element type (key, value) (line 43)
unicode_180240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 4), 'unicode', u'heliotrope')
unicode_180241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 18), 'unicode', u'#d94ff5')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180240, unicode_180241))
# Adding element type (key, value) (line 43)
unicode_180242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 4), 'unicode', u'yellow/green')
unicode_180243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 20), 'unicode', u'#c8fd3d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180242, unicode_180243))
# Adding element type (key, value) (line 43)
unicode_180244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 4), 'unicode', u'almost black')
unicode_180245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 20), 'unicode', u'#070d0d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180244, unicode_180245))
# Adding element type (key, value) (line 43)
unicode_180246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 4), 'unicode', u'cool blue')
unicode_180247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 17), 'unicode', u'#4984b8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180246, unicode_180247))
# Adding element type (key, value) (line 43)
unicode_180248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 4), 'unicode', u'leafy green')
unicode_180249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 19), 'unicode', u'#51b73b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180248, unicode_180249))
# Adding element type (key, value) (line 43)
unicode_180250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 4), 'unicode', u'mustard brown')
unicode_180251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 21), 'unicode', u'#ac7e04')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180250, unicode_180251))
# Adding element type (key, value) (line 43)
unicode_180252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 4), 'unicode', u'dusk')
unicode_180253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 12), 'unicode', u'#4e5481')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180252, unicode_180253))
# Adding element type (key, value) (line 43)
unicode_180254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 4), 'unicode', u'dull brown')
unicode_180255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 18), 'unicode', u'#876e4b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180254, unicode_180255))
# Adding element type (key, value) (line 43)
unicode_180256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 4), 'unicode', u'frog green')
unicode_180257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 18), 'unicode', u'#58bc08')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180256, unicode_180257))
# Adding element type (key, value) (line 43)
unicode_180258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 4), 'unicode', u'vivid green')
unicode_180259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 19), 'unicode', u'#2fef10')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180258, unicode_180259))
# Adding element type (key, value) (line 43)
unicode_180260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 4), 'unicode', u'bright light green')
unicode_180261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 26), 'unicode', u'#2dfe54')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180260, unicode_180261))
# Adding element type (key, value) (line 43)
unicode_180262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 4), 'unicode', u'fluro green')
unicode_180263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 19), 'unicode', u'#0aff02')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180262, unicode_180263))
# Adding element type (key, value) (line 43)
unicode_180264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 4), 'unicode', u'kiwi')
unicode_180265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 12), 'unicode', u'#9cef43')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180264, unicode_180265))
# Adding element type (key, value) (line 43)
unicode_180266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 4), 'unicode', u'seaweed')
unicode_180267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 15), 'unicode', u'#18d17b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180266, unicode_180267))
# Adding element type (key, value) (line 43)
unicode_180268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 4), 'unicode', u'navy green')
unicode_180269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 18), 'unicode', u'#35530a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180268, unicode_180269))
# Adding element type (key, value) (line 43)
unicode_180270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 4), 'unicode', u'ultramarine blue')
unicode_180271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 24), 'unicode', u'#1805db')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180270, unicode_180271))
# Adding element type (key, value) (line 43)
unicode_180272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 4), 'unicode', u'iris')
unicode_180273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 12), 'unicode', u'#6258c4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180272, unicode_180273))
# Adding element type (key, value) (line 43)
unicode_180274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 4), 'unicode', u'pastel orange')
unicode_180275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 21), 'unicode', u'#ff964f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180274, unicode_180275))
# Adding element type (key, value) (line 43)
unicode_180276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 4), 'unicode', u'yellowish orange')
unicode_180277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 24), 'unicode', u'#ffab0f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180276, unicode_180277))
# Adding element type (key, value) (line 43)
unicode_180278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 4), 'unicode', u'perrywinkle')
unicode_180279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 19), 'unicode', u'#8f8ce7')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180278, unicode_180279))
# Adding element type (key, value) (line 43)
unicode_180280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 4), 'unicode', u'tealish')
unicode_180281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 15), 'unicode', u'#24bca8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180280, unicode_180281))
# Adding element type (key, value) (line 43)
unicode_180282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 4), 'unicode', u'dark plum')
unicode_180283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 17), 'unicode', u'#3f012c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180282, unicode_180283))
# Adding element type (key, value) (line 43)
unicode_180284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 4), 'unicode', u'pear')
unicode_180285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 12), 'unicode', u'#cbf85f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180284, unicode_180285))
# Adding element type (key, value) (line 43)
unicode_180286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 4), 'unicode', u'pinkish orange')
unicode_180287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 22), 'unicode', u'#ff724c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180286, unicode_180287))
# Adding element type (key, value) (line 43)
unicode_180288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 4), 'unicode', u'midnight purple')
unicode_180289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 23), 'unicode', u'#280137')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180288, unicode_180289))
# Adding element type (key, value) (line 43)
unicode_180290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 4), 'unicode', u'light urple')
unicode_180291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 19), 'unicode', u'#b36ff6')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180290, unicode_180291))
# Adding element type (key, value) (line 43)
unicode_180292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 4), 'unicode', u'dark mint')
unicode_180293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 17), 'unicode', u'#48c072')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180292, unicode_180293))
# Adding element type (key, value) (line 43)
unicode_180294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 4), 'unicode', u'greenish tan')
unicode_180295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 20), 'unicode', u'#bccb7a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180294, unicode_180295))
# Adding element type (key, value) (line 43)
unicode_180296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 4), 'unicode', u'light burgundy')
unicode_180297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 22), 'unicode', u'#a8415b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180296, unicode_180297))
# Adding element type (key, value) (line 43)
unicode_180298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 4), 'unicode', u'turquoise blue')
unicode_180299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 22), 'unicode', u'#06b1c4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180298, unicode_180299))
# Adding element type (key, value) (line 43)
unicode_180300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 4), 'unicode', u'ugly pink')
unicode_180301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 17), 'unicode', u'#cd7584')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180300, unicode_180301))
# Adding element type (key, value) (line 43)
unicode_180302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 4), 'unicode', u'sandy')
unicode_180303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 13), 'unicode', u'#f1da7a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180302, unicode_180303))
# Adding element type (key, value) (line 43)
unicode_180304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 4), 'unicode', u'electric pink')
unicode_180305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 21), 'unicode', u'#ff0490')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180304, unicode_180305))
# Adding element type (key, value) (line 43)
unicode_180306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 4), 'unicode', u'muted purple')
unicode_180307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 20), 'unicode', u'#805b87')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180306, unicode_180307))
# Adding element type (key, value) (line 43)
unicode_180308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 4), 'unicode', u'mid green')
unicode_180309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 17), 'unicode', u'#50a747')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180308, unicode_180309))
# Adding element type (key, value) (line 43)
unicode_180310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 4), 'unicode', u'greyish')
unicode_180311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 15), 'unicode', u'#a8a495')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180310, unicode_180311))
# Adding element type (key, value) (line 43)
unicode_180312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 4), 'unicode', u'neon yellow')
unicode_180313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 19), 'unicode', u'#cfff04')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180312, unicode_180313))
# Adding element type (key, value) (line 43)
unicode_180314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 4), 'unicode', u'banana')
unicode_180315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 14), 'unicode', u'#ffff7e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180314, unicode_180315))
# Adding element type (key, value) (line 43)
unicode_180316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 4), 'unicode', u'carnation pink')
unicode_180317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 22), 'unicode', u'#ff7fa7')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180316, unicode_180317))
# Adding element type (key, value) (line 43)
unicode_180318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 4), 'unicode', u'tomato')
unicode_180319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 14), 'unicode', u'#ef4026')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180318, unicode_180319))
# Adding element type (key, value) (line 43)
unicode_180320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 4), 'unicode', u'sea')
unicode_180321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 11), 'unicode', u'#3c9992')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180320, unicode_180321))
# Adding element type (key, value) (line 43)
unicode_180322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 4), 'unicode', u'muddy brown')
unicode_180323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 19), 'unicode', u'#886806')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180322, unicode_180323))
# Adding element type (key, value) (line 43)
unicode_180324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 4), 'unicode', u'turquoise green')
unicode_180325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 23), 'unicode', u'#04f489')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180324, unicode_180325))
# Adding element type (key, value) (line 43)
unicode_180326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 4), 'unicode', u'buff')
unicode_180327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 12), 'unicode', u'#fef69e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180326, unicode_180327))
# Adding element type (key, value) (line 43)
unicode_180328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 4), 'unicode', u'fawn')
unicode_180329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 12), 'unicode', u'#cfaf7b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180328, unicode_180329))
# Adding element type (key, value) (line 43)
unicode_180330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 4), 'unicode', u'muted blue')
unicode_180331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 18), 'unicode', u'#3b719f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180330, unicode_180331))
# Adding element type (key, value) (line 43)
unicode_180332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 4), 'unicode', u'pale rose')
unicode_180333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 17), 'unicode', u'#fdc1c5')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180332, unicode_180333))
# Adding element type (key, value) (line 43)
unicode_180334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 4), 'unicode', u'dark mint green')
unicode_180335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 23), 'unicode', u'#20c073')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180334, unicode_180335))
# Adding element type (key, value) (line 43)
unicode_180336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 4), 'unicode', u'amethyst')
unicode_180337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 16), 'unicode', u'#9b5fc0')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180336, unicode_180337))
# Adding element type (key, value) (line 43)
unicode_180338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 4), 'unicode', u'blue/green')
unicode_180339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 18), 'unicode', u'#0f9b8e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180338, unicode_180339))
# Adding element type (key, value) (line 43)
unicode_180340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 4), 'unicode', u'chestnut')
unicode_180341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 16), 'unicode', u'#742802')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180340, unicode_180341))
# Adding element type (key, value) (line 43)
unicode_180342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 4), 'unicode', u'sick green')
unicode_180343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 18), 'unicode', u'#9db92c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180342, unicode_180343))
# Adding element type (key, value) (line 43)
unicode_180344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 4), 'unicode', u'pea')
unicode_180345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 11), 'unicode', u'#a4bf20')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180344, unicode_180345))
# Adding element type (key, value) (line 43)
unicode_180346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 4), 'unicode', u'rusty orange')
unicode_180347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 20), 'unicode', u'#cd5909')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180346, unicode_180347))
# Adding element type (key, value) (line 43)
unicode_180348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 4), 'unicode', u'stone')
unicode_180349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 13), 'unicode', u'#ada587')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180348, unicode_180349))
# Adding element type (key, value) (line 43)
unicode_180350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 4), 'unicode', u'rose red')
unicode_180351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 16), 'unicode', u'#be013c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180350, unicode_180351))
# Adding element type (key, value) (line 43)
unicode_180352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 4), 'unicode', u'pale aqua')
unicode_180353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 17), 'unicode', u'#b8ffeb')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180352, unicode_180353))
# Adding element type (key, value) (line 43)
unicode_180354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 4), 'unicode', u'deep orange')
unicode_180355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 19), 'unicode', u'#dc4d01')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180354, unicode_180355))
# Adding element type (key, value) (line 43)
unicode_180356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 4), 'unicode', u'earth')
unicode_180357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 13), 'unicode', u'#a2653e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180356, unicode_180357))
# Adding element type (key, value) (line 43)
unicode_180358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 4), 'unicode', u'mossy green')
unicode_180359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 19), 'unicode', u'#638b27')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180358, unicode_180359))
# Adding element type (key, value) (line 43)
unicode_180360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 4), 'unicode', u'grassy green')
unicode_180361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 20), 'unicode', u'#419c03')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180360, unicode_180361))
# Adding element type (key, value) (line 43)
unicode_180362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 4), 'unicode', u'pale lime green')
unicode_180363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 23), 'unicode', u'#b1ff65')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180362, unicode_180363))
# Adding element type (key, value) (line 43)
unicode_180364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 4), 'unicode', u'light grey blue')
unicode_180365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 23), 'unicode', u'#9dbcd4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180364, unicode_180365))
# Adding element type (key, value) (line 43)
unicode_180366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 4), 'unicode', u'pale grey')
unicode_180367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 17), 'unicode', u'#fdfdfe')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180366, unicode_180367))
# Adding element type (key, value) (line 43)
unicode_180368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 4), 'unicode', u'asparagus')
unicode_180369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 17), 'unicode', u'#77ab56')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180368, unicode_180369))
# Adding element type (key, value) (line 43)
unicode_180370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 4), 'unicode', u'blueberry')
unicode_180371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 17), 'unicode', u'#464196')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180370, unicode_180371))
# Adding element type (key, value) (line 43)
unicode_180372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 4), 'unicode', u'purple red')
unicode_180373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 18), 'unicode', u'#990147')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180372, unicode_180373))
# Adding element type (key, value) (line 43)
unicode_180374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 4), 'unicode', u'pale lime')
unicode_180375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 17), 'unicode', u'#befd73')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180374, unicode_180375))
# Adding element type (key, value) (line 43)
unicode_180376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 4), 'unicode', u'greenish teal')
unicode_180377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 21), 'unicode', u'#32bf84')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180376, unicode_180377))
# Adding element type (key, value) (line 43)
unicode_180378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 4), 'unicode', u'caramel')
unicode_180379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 15), 'unicode', u'#af6f09')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180378, unicode_180379))
# Adding element type (key, value) (line 43)
unicode_180380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 4), 'unicode', u'deep magenta')
unicode_180381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 20), 'unicode', u'#a0025c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180380, unicode_180381))
# Adding element type (key, value) (line 43)
unicode_180382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 4), 'unicode', u'light peach')
unicode_180383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 19), 'unicode', u'#ffd8b1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180382, unicode_180383))
# Adding element type (key, value) (line 43)
unicode_180384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 4), 'unicode', u'milk chocolate')
unicode_180385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 22), 'unicode', u'#7f4e1e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180384, unicode_180385))
# Adding element type (key, value) (line 43)
unicode_180386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 4), 'unicode', u'ocher')
unicode_180387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 13), 'unicode', u'#bf9b0c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180386, unicode_180387))
# Adding element type (key, value) (line 43)
unicode_180388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 4), 'unicode', u'off green')
unicode_180389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 17), 'unicode', u'#6ba353')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180388, unicode_180389))
# Adding element type (key, value) (line 43)
unicode_180390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 4), 'unicode', u'purply pink')
unicode_180391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 19), 'unicode', u'#f075e6')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180390, unicode_180391))
# Adding element type (key, value) (line 43)
unicode_180392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 4), 'unicode', u'lightblue')
unicode_180393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 17), 'unicode', u'#7bc8f6')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180392, unicode_180393))
# Adding element type (key, value) (line 43)
unicode_180394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 4), 'unicode', u'dusky blue')
unicode_180395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 18), 'unicode', u'#475f94')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180394, unicode_180395))
# Adding element type (key, value) (line 43)
unicode_180396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 4), 'unicode', u'golden')
unicode_180397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 14), 'unicode', u'#f5bf03')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180396, unicode_180397))
# Adding element type (key, value) (line 43)
unicode_180398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 4), 'unicode', u'light beige')
unicode_180399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 19), 'unicode', u'#fffeb6')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180398, unicode_180399))
# Adding element type (key, value) (line 43)
unicode_180400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 4), 'unicode', u'butter yellow')
unicode_180401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 21), 'unicode', u'#fffd74')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180400, unicode_180401))
# Adding element type (key, value) (line 43)
unicode_180402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 4), 'unicode', u'dusky purple')
unicode_180403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 20), 'unicode', u'#895b7b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180402, unicode_180403))
# Adding element type (key, value) (line 43)
unicode_180404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 4), 'unicode', u'french blue')
unicode_180405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 19), 'unicode', u'#436bad')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180404, unicode_180405))
# Adding element type (key, value) (line 43)
unicode_180406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 4), 'unicode', u'ugly yellow')
unicode_180407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 19), 'unicode', u'#d0c101')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180406, unicode_180407))
# Adding element type (key, value) (line 43)
unicode_180408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 4), 'unicode', u'greeny yellow')
unicode_180409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 21), 'unicode', u'#c6f808')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180408, unicode_180409))
# Adding element type (key, value) (line 43)
unicode_180410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 4), 'unicode', u'orangish red')
unicode_180411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 20), 'unicode', u'#f43605')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180410, unicode_180411))
# Adding element type (key, value) (line 43)
unicode_180412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 4), 'unicode', u'shamrock green')
unicode_180413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 22), 'unicode', u'#02c14d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180412, unicode_180413))
# Adding element type (key, value) (line 43)
unicode_180414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 4), 'unicode', u'orangish brown')
unicode_180415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 22), 'unicode', u'#b25f03')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180414, unicode_180415))
# Adding element type (key, value) (line 43)
unicode_180416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 4), 'unicode', u'tree green')
unicode_180417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 18), 'unicode', u'#2a7e19')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180416, unicode_180417))
# Adding element type (key, value) (line 43)
unicode_180418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 4), 'unicode', u'deep violet')
unicode_180419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 19), 'unicode', u'#490648')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180418, unicode_180419))
# Adding element type (key, value) (line 43)
unicode_180420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 4), 'unicode', u'gunmetal')
unicode_180421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 16), 'unicode', u'#536267')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180420, unicode_180421))
# Adding element type (key, value) (line 43)
unicode_180422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 4), 'unicode', u'blue/purple')
unicode_180423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 19), 'unicode', u'#5a06ef')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180422, unicode_180423))
# Adding element type (key, value) (line 43)
unicode_180424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 4), 'unicode', u'cherry')
unicode_180425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 14), 'unicode', u'#cf0234')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180424, unicode_180425))
# Adding element type (key, value) (line 43)
unicode_180426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 4), 'unicode', u'sandy brown')
unicode_180427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 19), 'unicode', u'#c4a661')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180426, unicode_180427))
# Adding element type (key, value) (line 43)
unicode_180428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 4), 'unicode', u'warm grey')
unicode_180429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 17), 'unicode', u'#978a84')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180428, unicode_180429))
# Adding element type (key, value) (line 43)
unicode_180430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 4), 'unicode', u'dark indigo')
unicode_180431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 19), 'unicode', u'#1f0954')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180430, unicode_180431))
# Adding element type (key, value) (line 43)
unicode_180432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 4), 'unicode', u'midnight')
unicode_180433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 16), 'unicode', u'#03012d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180432, unicode_180433))
# Adding element type (key, value) (line 43)
unicode_180434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 4), 'unicode', u'bluey green')
unicode_180435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 19), 'unicode', u'#2bb179')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180434, unicode_180435))
# Adding element type (key, value) (line 43)
unicode_180436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 4), 'unicode', u'grey pink')
unicode_180437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 17), 'unicode', u'#c3909b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180436, unicode_180437))
# Adding element type (key, value) (line 43)
unicode_180438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 4), 'unicode', u'soft purple')
unicode_180439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 19), 'unicode', u'#a66fb5')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180438, unicode_180439))
# Adding element type (key, value) (line 43)
unicode_180440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 4), 'unicode', u'blood')
unicode_180441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 13), 'unicode', u'#770001')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180440, unicode_180441))
# Adding element type (key, value) (line 43)
unicode_180442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 4), 'unicode', u'brown red')
unicode_180443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 17), 'unicode', u'#922b05')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180442, unicode_180443))
# Adding element type (key, value) (line 43)
unicode_180444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 4), 'unicode', u'medium grey')
unicode_180445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 19), 'unicode', u'#7d7f7c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180444, unicode_180445))
# Adding element type (key, value) (line 43)
unicode_180446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 4), 'unicode', u'berry')
unicode_180447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 13), 'unicode', u'#990f4b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180446, unicode_180447))
# Adding element type (key, value) (line 43)
unicode_180448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 4), 'unicode', u'poo')
unicode_180449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 11), 'unicode', u'#8f7303')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180448, unicode_180449))
# Adding element type (key, value) (line 43)
unicode_180450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 4), 'unicode', u'purpley pink')
unicode_180451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 20), 'unicode', u'#c83cb9')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180450, unicode_180451))
# Adding element type (key, value) (line 43)
unicode_180452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 4), 'unicode', u'light salmon')
unicode_180453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 20), 'unicode', u'#fea993')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180452, unicode_180453))
# Adding element type (key, value) (line 43)
unicode_180454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 4), 'unicode', u'snot')
unicode_180455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 12), 'unicode', u'#acbb0d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180454, unicode_180455))
# Adding element type (key, value) (line 43)
unicode_180456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 4), 'unicode', u'easter purple')
unicode_180457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 21), 'unicode', u'#c071fe')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180456, unicode_180457))
# Adding element type (key, value) (line 43)
unicode_180458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 4), 'unicode', u'light yellow green')
unicode_180459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 26), 'unicode', u'#ccfd7f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180458, unicode_180459))
# Adding element type (key, value) (line 43)
unicode_180460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 4), 'unicode', u'dark navy blue')
unicode_180461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 22), 'unicode', u'#00022e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180460, unicode_180461))
# Adding element type (key, value) (line 43)
unicode_180462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 4), 'unicode', u'drab')
unicode_180463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 12), 'unicode', u'#828344')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180462, unicode_180463))
# Adding element type (key, value) (line 43)
unicode_180464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 4), 'unicode', u'light rose')
unicode_180465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 18), 'unicode', u'#ffc5cb')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180464, unicode_180465))
# Adding element type (key, value) (line 43)
unicode_180466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 4), 'unicode', u'rouge')
unicode_180467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 13), 'unicode', u'#ab1239')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180466, unicode_180467))
# Adding element type (key, value) (line 43)
unicode_180468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 4), 'unicode', u'purplish red')
unicode_180469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 20), 'unicode', u'#b0054b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180468, unicode_180469))
# Adding element type (key, value) (line 43)
unicode_180470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 4), 'unicode', u'slime green')
unicode_180471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 19), 'unicode', u'#99cc04')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180470, unicode_180471))
# Adding element type (key, value) (line 43)
unicode_180472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 4), 'unicode', u'baby poop')
unicode_180473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 17), 'unicode', u'#937c00')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180472, unicode_180473))
# Adding element type (key, value) (line 43)
unicode_180474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 4), 'unicode', u'irish green')
unicode_180475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 19), 'unicode', u'#019529')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180474, unicode_180475))
# Adding element type (key, value) (line 43)
unicode_180476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 4), 'unicode', u'pink/purple')
unicode_180477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 19), 'unicode', u'#ef1de7')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180476, unicode_180477))
# Adding element type (key, value) (line 43)
unicode_180478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 4), 'unicode', u'dark navy')
unicode_180479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 17), 'unicode', u'#000435')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180478, unicode_180479))
# Adding element type (key, value) (line 43)
unicode_180480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 4), 'unicode', u'greeny blue')
unicode_180481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 19), 'unicode', u'#42b395')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180480, unicode_180481))
# Adding element type (key, value) (line 43)
unicode_180482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 4), 'unicode', u'light plum')
unicode_180483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 18), 'unicode', u'#9d5783')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180482, unicode_180483))
# Adding element type (key, value) (line 43)
unicode_180484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 4), 'unicode', u'pinkish grey')
unicode_180485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 20), 'unicode', u'#c8aca9')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180484, unicode_180485))
# Adding element type (key, value) (line 43)
unicode_180486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 4), 'unicode', u'dirty orange')
unicode_180487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 20), 'unicode', u'#c87606')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180486, unicode_180487))
# Adding element type (key, value) (line 43)
unicode_180488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 4), 'unicode', u'rust red')
unicode_180489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 16), 'unicode', u'#aa2704')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180488, unicode_180489))
# Adding element type (key, value) (line 43)
unicode_180490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 4), 'unicode', u'pale lilac')
unicode_180491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 18), 'unicode', u'#e4cbff')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180490, unicode_180491))
# Adding element type (key, value) (line 43)
unicode_180492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 4), 'unicode', u'orangey red')
unicode_180493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 19), 'unicode', u'#fa4224')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180492, unicode_180493))
# Adding element type (key, value) (line 43)
unicode_180494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 4), 'unicode', u'primary blue')
unicode_180495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 20), 'unicode', u'#0804f9')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180494, unicode_180495))
# Adding element type (key, value) (line 43)
unicode_180496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 4), 'unicode', u'kermit green')
unicode_180497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 20), 'unicode', u'#5cb200')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180496, unicode_180497))
# Adding element type (key, value) (line 43)
unicode_180498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 4), 'unicode', u'brownish purple')
unicode_180499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 23), 'unicode', u'#76424e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180498, unicode_180499))
# Adding element type (key, value) (line 43)
unicode_180500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 4), 'unicode', u'murky green')
unicode_180501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 19), 'unicode', u'#6c7a0e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180500, unicode_180501))
# Adding element type (key, value) (line 43)
unicode_180502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 4), 'unicode', u'wheat')
unicode_180503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 13), 'unicode', u'#fbdd7e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180502, unicode_180503))
# Adding element type (key, value) (line 43)
unicode_180504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 4), 'unicode', u'very dark purple')
unicode_180505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 24), 'unicode', u'#2a0134')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180504, unicode_180505))
# Adding element type (key, value) (line 43)
unicode_180506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 4), 'unicode', u'bottle green')
unicode_180507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 20), 'unicode', u'#044a05')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180506, unicode_180507))
# Adding element type (key, value) (line 43)
unicode_180508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 4), 'unicode', u'watermelon')
unicode_180509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 18), 'unicode', u'#fd4659')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180508, unicode_180509))
# Adding element type (key, value) (line 43)
unicode_180510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 4), 'unicode', u'deep sky blue')
unicode_180511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 21), 'unicode', u'#0d75f8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180510, unicode_180511))
# Adding element type (key, value) (line 43)
unicode_180512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 4), 'unicode', u'fire engine red')
unicode_180513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 23), 'unicode', u'#fe0002')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180512, unicode_180513))
# Adding element type (key, value) (line 43)
unicode_180514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 4), 'unicode', u'yellow ochre')
unicode_180515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 20), 'unicode', u'#cb9d06')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180514, unicode_180515))
# Adding element type (key, value) (line 43)
unicode_180516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 4), 'unicode', u'pumpkin orange')
unicode_180517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 22), 'unicode', u'#fb7d07')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180516, unicode_180517))
# Adding element type (key, value) (line 43)
unicode_180518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 4), 'unicode', u'pale olive')
unicode_180519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 18), 'unicode', u'#b9cc81')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180518, unicode_180519))
# Adding element type (key, value) (line 43)
unicode_180520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 4), 'unicode', u'light lilac')
unicode_180521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 19), 'unicode', u'#edc8ff')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180520, unicode_180521))
# Adding element type (key, value) (line 43)
unicode_180522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 4), 'unicode', u'lightish green')
unicode_180523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 22), 'unicode', u'#61e160')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180522, unicode_180523))
# Adding element type (key, value) (line 43)
unicode_180524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 4), 'unicode', u'carolina blue')
unicode_180525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 21), 'unicode', u'#8ab8fe')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180524, unicode_180525))
# Adding element type (key, value) (line 43)
unicode_180526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 4), 'unicode', u'mulberry')
unicode_180527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 16), 'unicode', u'#920a4e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180526, unicode_180527))
# Adding element type (key, value) (line 43)
unicode_180528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 4), 'unicode', u'shocking pink')
unicode_180529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 21), 'unicode', u'#fe02a2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180528, unicode_180529))
# Adding element type (key, value) (line 43)
unicode_180530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 4), 'unicode', u'auburn')
unicode_180531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 14), 'unicode', u'#9a3001')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180530, unicode_180531))
# Adding element type (key, value) (line 43)
unicode_180532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 4), 'unicode', u'bright lime green')
unicode_180533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 25), 'unicode', u'#65fe08')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180532, unicode_180533))
# Adding element type (key, value) (line 43)
unicode_180534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 4), 'unicode', u'celadon')
unicode_180535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 15), 'unicode', u'#befdb7')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180534, unicode_180535))
# Adding element type (key, value) (line 43)
unicode_180536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 4), 'unicode', u'pinkish brown')
unicode_180537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 21), 'unicode', u'#b17261')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180536, unicode_180537))
# Adding element type (key, value) (line 43)
unicode_180538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 4), 'unicode', u'poo brown')
unicode_180539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 17), 'unicode', u'#885f01')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180538, unicode_180539))
# Adding element type (key, value) (line 43)
unicode_180540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 4), 'unicode', u'bright sky blue')
unicode_180541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 23), 'unicode', u'#02ccfe')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180540, unicode_180541))
# Adding element type (key, value) (line 43)
unicode_180542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 4), 'unicode', u'celery')
unicode_180543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 14), 'unicode', u'#c1fd95')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180542, unicode_180543))
# Adding element type (key, value) (line 43)
unicode_180544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 4), 'unicode', u'dirt brown')
unicode_180545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 18), 'unicode', u'#836539')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180544, unicode_180545))
# Adding element type (key, value) (line 43)
unicode_180546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 4), 'unicode', u'strawberry')
unicode_180547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 18), 'unicode', u'#fb2943')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180546, unicode_180547))
# Adding element type (key, value) (line 43)
unicode_180548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 4), 'unicode', u'dark lime')
unicode_180549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 17), 'unicode', u'#84b701')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180548, unicode_180549))
# Adding element type (key, value) (line 43)
unicode_180550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 4), 'unicode', u'copper')
unicode_180551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 14), 'unicode', u'#b66325')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180550, unicode_180551))
# Adding element type (key, value) (line 43)
unicode_180552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 4), 'unicode', u'medium brown')
unicode_180553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 20), 'unicode', u'#7f5112')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180552, unicode_180553))
# Adding element type (key, value) (line 43)
unicode_180554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 4), 'unicode', u'muted green')
unicode_180555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 19), 'unicode', u'#5fa052')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180554, unicode_180555))
# Adding element type (key, value) (line 43)
unicode_180556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 4), 'unicode', u"robin's egg")
unicode_180557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 19), 'unicode', u'#6dedfd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180556, unicode_180557))
# Adding element type (key, value) (line 43)
unicode_180558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 4), 'unicode', u'bright aqua')
unicode_180559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 19), 'unicode', u'#0bf9ea')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180558, unicode_180559))
# Adding element type (key, value) (line 43)
unicode_180560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 4), 'unicode', u'bright lavender')
unicode_180561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 23), 'unicode', u'#c760ff')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180560, unicode_180561))
# Adding element type (key, value) (line 43)
unicode_180562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 4), 'unicode', u'ivory')
unicode_180563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 13), 'unicode', u'#ffffcb')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180562, unicode_180563))
# Adding element type (key, value) (line 43)
unicode_180564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 4), 'unicode', u'very light purple')
unicode_180565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 25), 'unicode', u'#f6cefc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180564, unicode_180565))
# Adding element type (key, value) (line 43)
unicode_180566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 4), 'unicode', u'light navy')
unicode_180567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 18), 'unicode', u'#155084')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180566, unicode_180567))
# Adding element type (key, value) (line 43)
unicode_180568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 4), 'unicode', u'pink red')
unicode_180569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 16), 'unicode', u'#f5054f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180568, unicode_180569))
# Adding element type (key, value) (line 43)
unicode_180570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 4), 'unicode', u'olive brown')
unicode_180571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 19), 'unicode', u'#645403')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180570, unicode_180571))
# Adding element type (key, value) (line 43)
unicode_180572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 4), 'unicode', u'poop brown')
unicode_180573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 18), 'unicode', u'#7a5901')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180572, unicode_180573))
# Adding element type (key, value) (line 43)
unicode_180574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 4), 'unicode', u'mustard green')
unicode_180575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 21), 'unicode', u'#a8b504')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180574, unicode_180575))
# Adding element type (key, value) (line 43)
unicode_180576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 4), 'unicode', u'ocean green')
unicode_180577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 19), 'unicode', u'#3d9973')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180576, unicode_180577))
# Adding element type (key, value) (line 43)
unicode_180578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 4), 'unicode', u'very dark blue')
unicode_180579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 22), 'unicode', u'#000133')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180578, unicode_180579))
# Adding element type (key, value) (line 43)
unicode_180580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 4), 'unicode', u'dusty green')
unicode_180581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 19), 'unicode', u'#76a973')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180580, unicode_180581))
# Adding element type (key, value) (line 43)
unicode_180582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 4), 'unicode', u'light navy blue')
unicode_180583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 23), 'unicode', u'#2e5a88')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180582, unicode_180583))
# Adding element type (key, value) (line 43)
unicode_180584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 4), 'unicode', u'minty green')
unicode_180585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 19), 'unicode', u'#0bf77d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180584, unicode_180585))
# Adding element type (key, value) (line 43)
unicode_180586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 4), 'unicode', u'adobe')
unicode_180587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 13), 'unicode', u'#bd6c48')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180586, unicode_180587))
# Adding element type (key, value) (line 43)
unicode_180588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 4), 'unicode', u'barney')
unicode_180589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 14), 'unicode', u'#ac1db8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180588, unicode_180589))
# Adding element type (key, value) (line 43)
unicode_180590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 4), 'unicode', u'jade green')
unicode_180591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 18), 'unicode', u'#2baf6a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180590, unicode_180591))
# Adding element type (key, value) (line 43)
unicode_180592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 4), 'unicode', u'bright light blue')
unicode_180593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 25), 'unicode', u'#26f7fd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180592, unicode_180593))
# Adding element type (key, value) (line 43)
unicode_180594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 4), 'unicode', u'light lime')
unicode_180595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 18), 'unicode', u'#aefd6c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180594, unicode_180595))
# Adding element type (key, value) (line 43)
unicode_180596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 4), 'unicode', u'dark khaki')
unicode_180597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 18), 'unicode', u'#9b8f55')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180596, unicode_180597))
# Adding element type (key, value) (line 43)
unicode_180598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 4), 'unicode', u'orange yellow')
unicode_180599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 21), 'unicode', u'#ffad01')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180598, unicode_180599))
# Adding element type (key, value) (line 43)
unicode_180600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 4), 'unicode', u'ocre')
unicode_180601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 12), 'unicode', u'#c69c04')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180600, unicode_180601))
# Adding element type (key, value) (line 43)
unicode_180602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 4), 'unicode', u'maize')
unicode_180603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 13), 'unicode', u'#f4d054')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180602, unicode_180603))
# Adding element type (key, value) (line 43)
unicode_180604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 4), 'unicode', u'faded pink')
unicode_180605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 18), 'unicode', u'#de9dac')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180604, unicode_180605))
# Adding element type (key, value) (line 43)
unicode_180606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 4), 'unicode', u'british racing green')
unicode_180607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 28), 'unicode', u'#05480d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180606, unicode_180607))
# Adding element type (key, value) (line 43)
unicode_180608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 4), 'unicode', u'sandstone')
unicode_180609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 17), 'unicode', u'#c9ae74')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180608, unicode_180609))
# Adding element type (key, value) (line 43)
unicode_180610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 4), 'unicode', u'mud brown')
unicode_180611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 17), 'unicode', u'#60460f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180610, unicode_180611))
# Adding element type (key, value) (line 43)
unicode_180612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 4), 'unicode', u'light sea green')
unicode_180613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 23), 'unicode', u'#98f6b0')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180612, unicode_180613))
# Adding element type (key, value) (line 43)
unicode_180614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 4), 'unicode', u'robin egg blue')
unicode_180615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 22), 'unicode', u'#8af1fe')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180614, unicode_180615))
# Adding element type (key, value) (line 43)
unicode_180616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 4), 'unicode', u'aqua marine')
unicode_180617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 19), 'unicode', u'#2ee8bb')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180616, unicode_180617))
# Adding element type (key, value) (line 43)
unicode_180618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 4), 'unicode', u'dark sea green')
unicode_180619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 22), 'unicode', u'#11875d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180618, unicode_180619))
# Adding element type (key, value) (line 43)
unicode_180620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 4), 'unicode', u'soft pink')
unicode_180621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 17), 'unicode', u'#fdb0c0')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180620, unicode_180621))
# Adding element type (key, value) (line 43)
unicode_180622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 4), 'unicode', u'orangey brown')
unicode_180623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 21), 'unicode', u'#b16002')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180622, unicode_180623))
# Adding element type (key, value) (line 43)
unicode_180624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 4), 'unicode', u'cherry red')
unicode_180625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 18), 'unicode', u'#f7022a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180624, unicode_180625))
# Adding element type (key, value) (line 43)
unicode_180626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 4), 'unicode', u'burnt yellow')
unicode_180627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 20), 'unicode', u'#d5ab09')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180626, unicode_180627))
# Adding element type (key, value) (line 43)
unicode_180628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 4), 'unicode', u'brownish grey')
unicode_180629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 21), 'unicode', u'#86775f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180628, unicode_180629))
# Adding element type (key, value) (line 43)
unicode_180630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 4), 'unicode', u'camel')
unicode_180631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 13), 'unicode', u'#c69f59')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180630, unicode_180631))
# Adding element type (key, value) (line 43)
unicode_180632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 4), 'unicode', u'purplish grey')
unicode_180633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 21), 'unicode', u'#7a687f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180632, unicode_180633))
# Adding element type (key, value) (line 43)
unicode_180634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 4), 'unicode', u'marine')
unicode_180635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 14), 'unicode', u'#042e60')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180634, unicode_180635))
# Adding element type (key, value) (line 43)
unicode_180636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 4), 'unicode', u'greyish pink')
unicode_180637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 20), 'unicode', u'#c88d94')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180636, unicode_180637))
# Adding element type (key, value) (line 43)
unicode_180638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 4), 'unicode', u'pale turquoise')
unicode_180639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 22), 'unicode', u'#a5fbd5')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180638, unicode_180639))
# Adding element type (key, value) (line 43)
unicode_180640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 4), 'unicode', u'pastel yellow')
unicode_180641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 21), 'unicode', u'#fffe71')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180640, unicode_180641))
# Adding element type (key, value) (line 43)
unicode_180642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 4), 'unicode', u'bluey purple')
unicode_180643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 20), 'unicode', u'#6241c7')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180642, unicode_180643))
# Adding element type (key, value) (line 43)
unicode_180644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 4), 'unicode', u'canary yellow')
unicode_180645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 21), 'unicode', u'#fffe40')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180644, unicode_180645))
# Adding element type (key, value) (line 43)
unicode_180646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 4), 'unicode', u'faded red')
unicode_180647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 17), 'unicode', u'#d3494e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180646, unicode_180647))
# Adding element type (key, value) (line 43)
unicode_180648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 4), 'unicode', u'sepia')
unicode_180649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 13), 'unicode', u'#985e2b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180648, unicode_180649))
# Adding element type (key, value) (line 43)
unicode_180650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 4), 'unicode', u'coffee')
unicode_180651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 14), 'unicode', u'#a6814c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180650, unicode_180651))
# Adding element type (key, value) (line 43)
unicode_180652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 4), 'unicode', u'bright magenta')
unicode_180653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 22), 'unicode', u'#ff08e8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180652, unicode_180653))
# Adding element type (key, value) (line 43)
unicode_180654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 4), 'unicode', u'mocha')
unicode_180655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 13), 'unicode', u'#9d7651')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180654, unicode_180655))
# Adding element type (key, value) (line 43)
unicode_180656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 4), 'unicode', u'ecru')
unicode_180657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 12), 'unicode', u'#feffca')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180656, unicode_180657))
# Adding element type (key, value) (line 43)
unicode_180658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 4), 'unicode', u'purpleish')
unicode_180659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 17), 'unicode', u'#98568d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180658, unicode_180659))
# Adding element type (key, value) (line 43)
unicode_180660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 4), 'unicode', u'cranberry')
unicode_180661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 17), 'unicode', u'#9e003a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180660, unicode_180661))
# Adding element type (key, value) (line 43)
unicode_180662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 4), 'unicode', u'darkish green')
unicode_180663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 21), 'unicode', u'#287c37')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180662, unicode_180663))
# Adding element type (key, value) (line 43)
unicode_180664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 4), 'unicode', u'brown orange')
unicode_180665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 20), 'unicode', u'#b96902')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180664, unicode_180665))
# Adding element type (key, value) (line 43)
unicode_180666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 4), 'unicode', u'dusky rose')
unicode_180667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 18), 'unicode', u'#ba6873')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180666, unicode_180667))
# Adding element type (key, value) (line 43)
unicode_180668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 4), 'unicode', u'melon')
unicode_180669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 13), 'unicode', u'#ff7855')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180668, unicode_180669))
# Adding element type (key, value) (line 43)
unicode_180670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 4), 'unicode', u'sickly green')
unicode_180671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 20), 'unicode', u'#94b21c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180670, unicode_180671))
# Adding element type (key, value) (line 43)
unicode_180672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 4), 'unicode', u'silver')
unicode_180673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 14), 'unicode', u'#c5c9c7')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180672, unicode_180673))
# Adding element type (key, value) (line 43)
unicode_180674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 4), 'unicode', u'purply blue')
unicode_180675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 19), 'unicode', u'#661aee')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180674, unicode_180675))
# Adding element type (key, value) (line 43)
unicode_180676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 4), 'unicode', u'purpleish blue')
unicode_180677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 22), 'unicode', u'#6140ef')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180676, unicode_180677))
# Adding element type (key, value) (line 43)
unicode_180678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 4), 'unicode', u'hospital green')
unicode_180679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 22), 'unicode', u'#9be5aa')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180678, unicode_180679))
# Adding element type (key, value) (line 43)
unicode_180680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 4), 'unicode', u'shit brown')
unicode_180681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 18), 'unicode', u'#7b5804')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180680, unicode_180681))
# Adding element type (key, value) (line 43)
unicode_180682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 4), 'unicode', u'mid blue')
unicode_180683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 16), 'unicode', u'#276ab3')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180682, unicode_180683))
# Adding element type (key, value) (line 43)
unicode_180684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 4), 'unicode', u'amber')
unicode_180685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 13), 'unicode', u'#feb308')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180684, unicode_180685))
# Adding element type (key, value) (line 43)
unicode_180686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 4), 'unicode', u'easter green')
unicode_180687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 20), 'unicode', u'#8cfd7e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180686, unicode_180687))
# Adding element type (key, value) (line 43)
unicode_180688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 4), 'unicode', u'soft blue')
unicode_180689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 17), 'unicode', u'#6488ea')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180688, unicode_180689))
# Adding element type (key, value) (line 43)
unicode_180690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 4), 'unicode', u'cerulean blue')
unicode_180691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 21), 'unicode', u'#056eee')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180690, unicode_180691))
# Adding element type (key, value) (line 43)
unicode_180692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 4), 'unicode', u'golden brown')
unicode_180693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 20), 'unicode', u'#b27a01')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180692, unicode_180693))
# Adding element type (key, value) (line 43)
unicode_180694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 4), 'unicode', u'bright turquoise')
unicode_180695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 24), 'unicode', u'#0ffef9')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180694, unicode_180695))
# Adding element type (key, value) (line 43)
unicode_180696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 4), 'unicode', u'red pink')
unicode_180697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 16), 'unicode', u'#fa2a55')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180696, unicode_180697))
# Adding element type (key, value) (line 43)
unicode_180698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 4), 'unicode', u'red purple')
unicode_180699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 18), 'unicode', u'#820747')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180698, unicode_180699))
# Adding element type (key, value) (line 43)
unicode_180700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 4), 'unicode', u'greyish brown')
unicode_180701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 21), 'unicode', u'#7a6a4f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180700, unicode_180701))
# Adding element type (key, value) (line 43)
unicode_180702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 4), 'unicode', u'vermillion')
unicode_180703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 18), 'unicode', u'#f4320c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180702, unicode_180703))
# Adding element type (key, value) (line 43)
unicode_180704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 4), 'unicode', u'russet')
unicode_180705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 14), 'unicode', u'#a13905')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180704, unicode_180705))
# Adding element type (key, value) (line 43)
unicode_180706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 4), 'unicode', u'steel grey')
unicode_180707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 18), 'unicode', u'#6f828a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180706, unicode_180707))
# Adding element type (key, value) (line 43)
unicode_180708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 4), 'unicode', u'lighter purple')
unicode_180709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 22), 'unicode', u'#a55af4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180708, unicode_180709))
# Adding element type (key, value) (line 43)
unicode_180710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 4), 'unicode', u'bright violet')
unicode_180711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 21), 'unicode', u'#ad0afd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180710, unicode_180711))
# Adding element type (key, value) (line 43)
unicode_180712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 4), 'unicode', u'prussian blue')
unicode_180713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 21), 'unicode', u'#004577')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180712, unicode_180713))
# Adding element type (key, value) (line 43)
unicode_180714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 4), 'unicode', u'slate green')
unicode_180715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 19), 'unicode', u'#658d6d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180714, unicode_180715))
# Adding element type (key, value) (line 43)
unicode_180716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 4), 'unicode', u'dirty pink')
unicode_180717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 18), 'unicode', u'#ca7b80')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180716, unicode_180717))
# Adding element type (key, value) (line 43)
unicode_180718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 4), 'unicode', u'dark blue green')
unicode_180719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 23), 'unicode', u'#005249')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180718, unicode_180719))
# Adding element type (key, value) (line 43)
unicode_180720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 4), 'unicode', u'pine')
unicode_180721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 12), 'unicode', u'#2b5d34')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180720, unicode_180721))
# Adding element type (key, value) (line 43)
unicode_180722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 4), 'unicode', u'yellowy green')
unicode_180723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 21), 'unicode', u'#bff128')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180722, unicode_180723))
# Adding element type (key, value) (line 43)
unicode_180724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 4), 'unicode', u'dark gold')
unicode_180725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 17), 'unicode', u'#b59410')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180724, unicode_180725))
# Adding element type (key, value) (line 43)
unicode_180726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 4), 'unicode', u'bluish')
unicode_180727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 14), 'unicode', u'#2976bb')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180726, unicode_180727))
# Adding element type (key, value) (line 43)
unicode_180728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 4), 'unicode', u'darkish blue')
unicode_180729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 20), 'unicode', u'#014182')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180728, unicode_180729))
# Adding element type (key, value) (line 43)
unicode_180730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 4), 'unicode', u'dull red')
unicode_180731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 16), 'unicode', u'#bb3f3f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180730, unicode_180731))
# Adding element type (key, value) (line 43)
unicode_180732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 4), 'unicode', u'pinky red')
unicode_180733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 17), 'unicode', u'#fc2647')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180732, unicode_180733))
# Adding element type (key, value) (line 43)
unicode_180734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 4), 'unicode', u'bronze')
unicode_180735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 14), 'unicode', u'#a87900')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180734, unicode_180735))
# Adding element type (key, value) (line 43)
unicode_180736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 4), 'unicode', u'pale teal')
unicode_180737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 17), 'unicode', u'#82cbb2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180736, unicode_180737))
# Adding element type (key, value) (line 43)
unicode_180738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 4), 'unicode', u'military green')
unicode_180739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 22), 'unicode', u'#667c3e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180738, unicode_180739))
# Adding element type (key, value) (line 43)
unicode_180740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 4), 'unicode', u'barbie pink')
unicode_180741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 19), 'unicode', u'#fe46a5')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180740, unicode_180741))
# Adding element type (key, value) (line 43)
unicode_180742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 4), 'unicode', u'bubblegum pink')
unicode_180743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 22), 'unicode', u'#fe83cc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180742, unicode_180743))
# Adding element type (key, value) (line 43)
unicode_180744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 4), 'unicode', u'pea soup green')
unicode_180745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 22), 'unicode', u'#94a617')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180744, unicode_180745))
# Adding element type (key, value) (line 43)
unicode_180746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 4), 'unicode', u'dark mustard')
unicode_180747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 20), 'unicode', u'#a88905')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180746, unicode_180747))
# Adding element type (key, value) (line 43)
unicode_180748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 4), 'unicode', u'shit')
unicode_180749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 12), 'unicode', u'#7f5f00')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180748, unicode_180749))
# Adding element type (key, value) (line 43)
unicode_180750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 642, 4), 'unicode', u'medium purple')
unicode_180751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 642, 21), 'unicode', u'#9e43a2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180750, unicode_180751))
# Adding element type (key, value) (line 43)
unicode_180752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 4), 'unicode', u'very dark green')
unicode_180753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 23), 'unicode', u'#062e03')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180752, unicode_180753))
# Adding element type (key, value) (line 43)
unicode_180754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 644, 4), 'unicode', u'dirt')
unicode_180755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 644, 12), 'unicode', u'#8a6e45')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180754, unicode_180755))
# Adding element type (key, value) (line 43)
unicode_180756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 645, 4), 'unicode', u'dusky pink')
unicode_180757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 645, 18), 'unicode', u'#cc7a8b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180756, unicode_180757))
# Adding element type (key, value) (line 43)
unicode_180758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 4), 'unicode', u'red violet')
unicode_180759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 18), 'unicode', u'#9e0168')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180758, unicode_180759))
# Adding element type (key, value) (line 43)
unicode_180760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 647, 4), 'unicode', u'lemon yellow')
unicode_180761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 647, 20), 'unicode', u'#fdff38')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180760, unicode_180761))
# Adding element type (key, value) (line 43)
unicode_180762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 648, 4), 'unicode', u'pistachio')
unicode_180763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 648, 17), 'unicode', u'#c0fa8b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180762, unicode_180763))
# Adding element type (key, value) (line 43)
unicode_180764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 4), 'unicode', u'dull yellow')
unicode_180765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 19), 'unicode', u'#eedc5b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180764, unicode_180765))
# Adding element type (key, value) (line 43)
unicode_180766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 4), 'unicode', u'dark lime green')
unicode_180767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 23), 'unicode', u'#7ebd01')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180766, unicode_180767))
# Adding element type (key, value) (line 43)
unicode_180768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 651, 4), 'unicode', u'denim blue')
unicode_180769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 651, 18), 'unicode', u'#3b5b92')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180768, unicode_180769))
# Adding element type (key, value) (line 43)
unicode_180770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 4), 'unicode', u'teal blue')
unicode_180771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 17), 'unicode', u'#01889f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180770, unicode_180771))
# Adding element type (key, value) (line 43)
unicode_180772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 4), 'unicode', u'lightish blue')
unicode_180773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 21), 'unicode', u'#3d7afd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180772, unicode_180773))
# Adding element type (key, value) (line 43)
unicode_180774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 4), 'unicode', u'purpley blue')
unicode_180775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 20), 'unicode', u'#5f34e7')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180774, unicode_180775))
# Adding element type (key, value) (line 43)
unicode_180776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 4), 'unicode', u'light indigo')
unicode_180777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 20), 'unicode', u'#6d5acf')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180776, unicode_180777))
# Adding element type (key, value) (line 43)
unicode_180778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 4), 'unicode', u'swamp green')
unicode_180779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 19), 'unicode', u'#748500')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180778, unicode_180779))
# Adding element type (key, value) (line 43)
unicode_180780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 4), 'unicode', u'brown green')
unicode_180781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 19), 'unicode', u'#706c11')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180780, unicode_180781))
# Adding element type (key, value) (line 43)
unicode_180782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 4), 'unicode', u'dark maroon')
unicode_180783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 19), 'unicode', u'#3c0008')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180782, unicode_180783))
# Adding element type (key, value) (line 43)
unicode_180784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 4), 'unicode', u'hot purple')
unicode_180785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 18), 'unicode', u'#cb00f5')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180784, unicode_180785))
# Adding element type (key, value) (line 43)
unicode_180786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 4), 'unicode', u'dark forest green')
unicode_180787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 25), 'unicode', u'#002d04')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180786, unicode_180787))
# Adding element type (key, value) (line 43)
unicode_180788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 4), 'unicode', u'faded blue')
unicode_180789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 18), 'unicode', u'#658cbb')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180788, unicode_180789))
# Adding element type (key, value) (line 43)
unicode_180790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 4), 'unicode', u'drab green')
unicode_180791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 18), 'unicode', u'#749551')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180790, unicode_180791))
# Adding element type (key, value) (line 43)
unicode_180792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 4), 'unicode', u'light lime green')
unicode_180793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 24), 'unicode', u'#b9ff66')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180792, unicode_180793))
# Adding element type (key, value) (line 43)
unicode_180794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 4), 'unicode', u'snot green')
unicode_180795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 18), 'unicode', u'#9dc100')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180794, unicode_180795))
# Adding element type (key, value) (line 43)
unicode_180796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, 4), 'unicode', u'yellowish')
unicode_180797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, 17), 'unicode', u'#faee66')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180796, unicode_180797))
# Adding element type (key, value) (line 43)
unicode_180798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 4), 'unicode', u'light blue green')
unicode_180799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 24), 'unicode', u'#7efbb3')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180798, unicode_180799))
# Adding element type (key, value) (line 43)
unicode_180800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 4), 'unicode', u'bordeaux')
unicode_180801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 16), 'unicode', u'#7b002c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180800, unicode_180801))
# Adding element type (key, value) (line 43)
unicode_180802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 4), 'unicode', u'light mauve')
unicode_180803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 19), 'unicode', u'#c292a1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180802, unicode_180803))
# Adding element type (key, value) (line 43)
unicode_180804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 4), 'unicode', u'ocean')
unicode_180805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 13), 'unicode', u'#017b92')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180804, unicode_180805))
# Adding element type (key, value) (line 43)
unicode_180806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 670, 4), 'unicode', u'marigold')
unicode_180807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 670, 16), 'unicode', u'#fcc006')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180806, unicode_180807))
# Adding element type (key, value) (line 43)
unicode_180808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 4), 'unicode', u'muddy green')
unicode_180809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 19), 'unicode', u'#657432')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180808, unicode_180809))
# Adding element type (key, value) (line 43)
unicode_180810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 4), 'unicode', u'dull orange')
unicode_180811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 19), 'unicode', u'#d8863b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180810, unicode_180811))
# Adding element type (key, value) (line 43)
unicode_180812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 4), 'unicode', u'steel')
unicode_180813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 13), 'unicode', u'#738595')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180812, unicode_180813))
# Adding element type (key, value) (line 43)
unicode_180814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 4), 'unicode', u'electric purple')
unicode_180815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 23), 'unicode', u'#aa23ff')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180814, unicode_180815))
# Adding element type (key, value) (line 43)
unicode_180816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 4), 'unicode', u'fluorescent green')
unicode_180817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 25), 'unicode', u'#08ff08')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180816, unicode_180817))
# Adding element type (key, value) (line 43)
unicode_180818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 4), 'unicode', u'yellowish brown')
unicode_180819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 23), 'unicode', u'#9b7a01')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180818, unicode_180819))
# Adding element type (key, value) (line 43)
unicode_180820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 4), 'unicode', u'blush')
unicode_180821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 13), 'unicode', u'#f29e8e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180820, unicode_180821))
# Adding element type (key, value) (line 43)
unicode_180822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 4), 'unicode', u'soft green')
unicode_180823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 18), 'unicode', u'#6fc276')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180822, unicode_180823))
# Adding element type (key, value) (line 43)
unicode_180824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 4), 'unicode', u'bright orange')
unicode_180825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 21), 'unicode', u'#ff5b00')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180824, unicode_180825))
# Adding element type (key, value) (line 43)
unicode_180826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 4), 'unicode', u'lemon')
unicode_180827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 13), 'unicode', u'#fdff52')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180826, unicode_180827))
# Adding element type (key, value) (line 43)
unicode_180828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 4), 'unicode', u'purple grey')
unicode_180829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 19), 'unicode', u'#866f85')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180828, unicode_180829))
# Adding element type (key, value) (line 43)
unicode_180830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 4), 'unicode', u'acid green')
unicode_180831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 18), 'unicode', u'#8ffe09')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180830, unicode_180831))
# Adding element type (key, value) (line 43)
unicode_180832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 4), 'unicode', u'pale lavender')
unicode_180833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 21), 'unicode', u'#eecffe')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180832, unicode_180833))
# Adding element type (key, value) (line 43)
unicode_180834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 4), 'unicode', u'violet blue')
unicode_180835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 19), 'unicode', u'#510ac9')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180834, unicode_180835))
# Adding element type (key, value) (line 43)
unicode_180836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 4), 'unicode', u'light forest green')
unicode_180837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 26), 'unicode', u'#4f9153')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180836, unicode_180837))
# Adding element type (key, value) (line 43)
unicode_180838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 4), 'unicode', u'burnt red')
unicode_180839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 17), 'unicode', u'#9f2305')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180838, unicode_180839))
# Adding element type (key, value) (line 43)
unicode_180840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 4), 'unicode', u'khaki green')
unicode_180841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 19), 'unicode', u'#728639')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180840, unicode_180841))
# Adding element type (key, value) (line 43)
unicode_180842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 4), 'unicode', u'cerise')
unicode_180843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 14), 'unicode', u'#de0c62')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180842, unicode_180843))
# Adding element type (key, value) (line 43)
unicode_180844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 4), 'unicode', u'faded purple')
unicode_180845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 20), 'unicode', u'#916e99')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180844, unicode_180845))
# Adding element type (key, value) (line 43)
unicode_180846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 4), 'unicode', u'apricot')
unicode_180847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 15), 'unicode', u'#ffb16d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180846, unicode_180847))
# Adding element type (key, value) (line 43)
unicode_180848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 4), 'unicode', u'dark olive green')
unicode_180849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 24), 'unicode', u'#3c4d03')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180848, unicode_180849))
# Adding element type (key, value) (line 43)
unicode_180850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 4), 'unicode', u'grey brown')
unicode_180851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 18), 'unicode', u'#7f7053')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180850, unicode_180851))
# Adding element type (key, value) (line 43)
unicode_180852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 4), 'unicode', u'green grey')
unicode_180853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 18), 'unicode', u'#77926f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180852, unicode_180853))
# Adding element type (key, value) (line 43)
unicode_180854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 4), 'unicode', u'true blue')
unicode_180855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 17), 'unicode', u'#010fcc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180854, unicode_180855))
# Adding element type (key, value) (line 43)
unicode_180856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 4), 'unicode', u'pale violet')
unicode_180857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 19), 'unicode', u'#ceaefa')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180856, unicode_180857))
# Adding element type (key, value) (line 43)
unicode_180858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 4), 'unicode', u'periwinkle blue')
unicode_180859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 23), 'unicode', u'#8f99fb')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180858, unicode_180859))
# Adding element type (key, value) (line 43)
unicode_180860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 4), 'unicode', u'light sky blue')
unicode_180861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 22), 'unicode', u'#c6fcff')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180860, unicode_180861))
# Adding element type (key, value) (line 43)
unicode_180862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 4), 'unicode', u'blurple')
unicode_180863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 15), 'unicode', u'#5539cc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180862, unicode_180863))
# Adding element type (key, value) (line 43)
unicode_180864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 4), 'unicode', u'green brown')
unicode_180865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 19), 'unicode', u'#544e03')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180864, unicode_180865))
# Adding element type (key, value) (line 43)
unicode_180866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 4), 'unicode', u'bluegreen')
unicode_180867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 17), 'unicode', u'#017a79')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180866, unicode_180867))
# Adding element type (key, value) (line 43)
unicode_180868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 4), 'unicode', u'bright teal')
unicode_180869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 19), 'unicode', u'#01f9c6')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180868, unicode_180869))
# Adding element type (key, value) (line 43)
unicode_180870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 4), 'unicode', u'brownish yellow')
unicode_180871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 23), 'unicode', u'#c9b003')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180870, unicode_180871))
# Adding element type (key, value) (line 43)
unicode_180872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 703, 4), 'unicode', u'pea soup')
unicode_180873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 703, 16), 'unicode', u'#929901')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180872, unicode_180873))
# Adding element type (key, value) (line 43)
unicode_180874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 4), 'unicode', u'forest')
unicode_180875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 14), 'unicode', u'#0b5509')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180874, unicode_180875))
# Adding element type (key, value) (line 43)
unicode_180876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 4), 'unicode', u'barney purple')
unicode_180877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 21), 'unicode', u'#a00498')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180876, unicode_180877))
# Adding element type (key, value) (line 43)
unicode_180878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 4), 'unicode', u'ultramarine')
unicode_180879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 19), 'unicode', u'#2000b1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180878, unicode_180879))
# Adding element type (key, value) (line 43)
unicode_180880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 4), 'unicode', u'purplish')
unicode_180881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 16), 'unicode', u'#94568c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180880, unicode_180881))
# Adding element type (key, value) (line 43)
unicode_180882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 4), 'unicode', u'puke yellow')
unicode_180883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 19), 'unicode', u'#c2be0e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180882, unicode_180883))
# Adding element type (key, value) (line 43)
unicode_180884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 4), 'unicode', u'bluish grey')
unicode_180885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 19), 'unicode', u'#748b97')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180884, unicode_180885))
# Adding element type (key, value) (line 43)
unicode_180886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 4), 'unicode', u'dark periwinkle')
unicode_180887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 23), 'unicode', u'#665fd1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180886, unicode_180887))
# Adding element type (key, value) (line 43)
unicode_180888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 4), 'unicode', u'dark lilac')
unicode_180889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 18), 'unicode', u'#9c6da5')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180888, unicode_180889))
# Adding element type (key, value) (line 43)
unicode_180890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 4), 'unicode', u'reddish')
unicode_180891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 15), 'unicode', u'#c44240')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180890, unicode_180891))
# Adding element type (key, value) (line 43)
unicode_180892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 4), 'unicode', u'light maroon')
unicode_180893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 20), 'unicode', u'#a24857')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180892, unicode_180893))
# Adding element type (key, value) (line 43)
unicode_180894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 714, 4), 'unicode', u'dusty purple')
unicode_180895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 714, 20), 'unicode', u'#825f87')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180894, unicode_180895))
# Adding element type (key, value) (line 43)
unicode_180896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 4), 'unicode', u'terra cotta')
unicode_180897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 19), 'unicode', u'#c9643b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180896, unicode_180897))
# Adding element type (key, value) (line 43)
unicode_180898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 4), 'unicode', u'avocado')
unicode_180899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 15), 'unicode', u'#90b134')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180898, unicode_180899))
# Adding element type (key, value) (line 43)
unicode_180900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 717, 4), 'unicode', u'marine blue')
unicode_180901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 717, 19), 'unicode', u'#01386a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180900, unicode_180901))
# Adding element type (key, value) (line 43)
unicode_180902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 4), 'unicode', u'teal green')
unicode_180903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 18), 'unicode', u'#25a36f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180902, unicode_180903))
# Adding element type (key, value) (line 43)
unicode_180904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 4), 'unicode', u'slate grey')
unicode_180905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 18), 'unicode', u'#59656d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180904, unicode_180905))
# Adding element type (key, value) (line 43)
unicode_180906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 720, 4), 'unicode', u'lighter green')
unicode_180907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 720, 21), 'unicode', u'#75fd63')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180906, unicode_180907))
# Adding element type (key, value) (line 43)
unicode_180908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, 4), 'unicode', u'electric green')
unicode_180909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, 22), 'unicode', u'#21fc0d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180908, unicode_180909))
# Adding element type (key, value) (line 43)
unicode_180910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 4), 'unicode', u'dusty blue')
unicode_180911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 18), 'unicode', u'#5a86ad')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180910, unicode_180911))
# Adding element type (key, value) (line 43)
unicode_180912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 723, 4), 'unicode', u'golden yellow')
unicode_180913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 723, 21), 'unicode', u'#fec615')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180912, unicode_180913))
# Adding element type (key, value) (line 43)
unicode_180914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 4), 'unicode', u'bright yellow')
unicode_180915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 21), 'unicode', u'#fffd01')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180914, unicode_180915))
# Adding element type (key, value) (line 43)
unicode_180916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 4), 'unicode', u'light lavender')
unicode_180917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 22), 'unicode', u'#dfc5fe')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180916, unicode_180917))
# Adding element type (key, value) (line 43)
unicode_180918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 4), 'unicode', u'umber')
unicode_180919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 13), 'unicode', u'#b26400')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180918, unicode_180919))
# Adding element type (key, value) (line 43)
unicode_180920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 4), 'unicode', u'poop')
unicode_180921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 12), 'unicode', u'#7f5e00')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180920, unicode_180921))
# Adding element type (key, value) (line 43)
unicode_180922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 4), 'unicode', u'dark peach')
unicode_180923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 18), 'unicode', u'#de7e5d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180922, unicode_180923))
# Adding element type (key, value) (line 43)
unicode_180924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 4), 'unicode', u'jungle green')
unicode_180925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 20), 'unicode', u'#048243')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180924, unicode_180925))
# Adding element type (key, value) (line 43)
unicode_180926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 730, 4), 'unicode', u'eggshell')
unicode_180927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 730, 16), 'unicode', u'#ffffd4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180926, unicode_180927))
# Adding element type (key, value) (line 43)
unicode_180928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 731, 4), 'unicode', u'denim')
unicode_180929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 731, 13), 'unicode', u'#3b638c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180928, unicode_180929))
# Adding element type (key, value) (line 43)
unicode_180930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 4), 'unicode', u'yellow brown')
unicode_180931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 20), 'unicode', u'#b79400')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180930, unicode_180931))
# Adding element type (key, value) (line 43)
unicode_180932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 4), 'unicode', u'dull purple')
unicode_180933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 19), 'unicode', u'#84597e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180932, unicode_180933))
# Adding element type (key, value) (line 43)
unicode_180934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 4), 'unicode', u'chocolate brown')
unicode_180935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 23), 'unicode', u'#411900')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180934, unicode_180935))
# Adding element type (key, value) (line 43)
unicode_180936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 735, 4), 'unicode', u'wine red')
unicode_180937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 735, 16), 'unicode', u'#7b0323')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180936, unicode_180937))
# Adding element type (key, value) (line 43)
unicode_180938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 4), 'unicode', u'neon blue')
unicode_180939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 17), 'unicode', u'#04d9ff')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180938, unicode_180939))
# Adding element type (key, value) (line 43)
unicode_180940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 737, 4), 'unicode', u'dirty green')
unicode_180941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 737, 19), 'unicode', u'#667e2c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180940, unicode_180941))
# Adding element type (key, value) (line 43)
unicode_180942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 4), 'unicode', u'light tan')
unicode_180943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 17), 'unicode', u'#fbeeac')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180942, unicode_180943))
# Adding element type (key, value) (line 43)
unicode_180944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 4), 'unicode', u'ice blue')
unicode_180945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 16), 'unicode', u'#d7fffe')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180944, unicode_180945))
# Adding element type (key, value) (line 43)
unicode_180946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 4), 'unicode', u'cadet blue')
unicode_180947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 18), 'unicode', u'#4e7496')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180946, unicode_180947))
# Adding element type (key, value) (line 43)
unicode_180948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 4), 'unicode', u'dark mauve')
unicode_180949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 18), 'unicode', u'#874c62')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180948, unicode_180949))
# Adding element type (key, value) (line 43)
unicode_180950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 4), 'unicode', u'very light blue')
unicode_180951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 23), 'unicode', u'#d5ffff')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180950, unicode_180951))
# Adding element type (key, value) (line 43)
unicode_180952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 4), 'unicode', u'grey purple')
unicode_180953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 19), 'unicode', u'#826d8c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180952, unicode_180953))
# Adding element type (key, value) (line 43)
unicode_180954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 4), 'unicode', u'pastel pink')
unicode_180955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 19), 'unicode', u'#ffbacd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180954, unicode_180955))
# Adding element type (key, value) (line 43)
unicode_180956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 4), 'unicode', u'very light green')
unicode_180957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 24), 'unicode', u'#d1ffbd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180956, unicode_180957))
# Adding element type (key, value) (line 43)
unicode_180958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 4), 'unicode', u'dark sky blue')
unicode_180959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 21), 'unicode', u'#448ee4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180958, unicode_180959))
# Adding element type (key, value) (line 43)
unicode_180960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 4), 'unicode', u'evergreen')
unicode_180961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 17), 'unicode', u'#05472a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180960, unicode_180961))
# Adding element type (key, value) (line 43)
unicode_180962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 4), 'unicode', u'dull pink')
unicode_180963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 17), 'unicode', u'#d5869d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180962, unicode_180963))
# Adding element type (key, value) (line 43)
unicode_180964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 4), 'unicode', u'aubergine')
unicode_180965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 17), 'unicode', u'#3d0734')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180964, unicode_180965))
# Adding element type (key, value) (line 43)
unicode_180966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 4), 'unicode', u'mahogany')
unicode_180967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 16), 'unicode', u'#4a0100')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180966, unicode_180967))
# Adding element type (key, value) (line 43)
unicode_180968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 4), 'unicode', u'reddish orange')
unicode_180969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 22), 'unicode', u'#f8481c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180968, unicode_180969))
# Adding element type (key, value) (line 43)
unicode_180970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 752, 4), 'unicode', u'deep green')
unicode_180971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 752, 18), 'unicode', u'#02590f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180970, unicode_180971))
# Adding element type (key, value) (line 43)
unicode_180972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 753, 4), 'unicode', u'vomit green')
unicode_180973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 753, 19), 'unicode', u'#89a203')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180972, unicode_180973))
# Adding element type (key, value) (line 43)
unicode_180974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 4), 'unicode', u'purple pink')
unicode_180975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, 19), 'unicode', u'#e03fd8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180974, unicode_180975))
# Adding element type (key, value) (line 43)
unicode_180976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 4), 'unicode', u'dusty pink')
unicode_180977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 18), 'unicode', u'#d58a94')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180976, unicode_180977))
# Adding element type (key, value) (line 43)
unicode_180978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 4), 'unicode', u'faded green')
unicode_180979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 19), 'unicode', u'#7bb274')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180978, unicode_180979))
# Adding element type (key, value) (line 43)
unicode_180980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 4), 'unicode', u'camo green')
unicode_180981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 18), 'unicode', u'#526525')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180980, unicode_180981))
# Adding element type (key, value) (line 43)
unicode_180982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 4), 'unicode', u'pinky purple')
unicode_180983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 20), 'unicode', u'#c94cbe')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180982, unicode_180983))
# Adding element type (key, value) (line 43)
unicode_180984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 4), 'unicode', u'pink purple')
unicode_180985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 19), 'unicode', u'#db4bda')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180984, unicode_180985))
# Adding element type (key, value) (line 43)
unicode_180986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 4), 'unicode', u'brownish red')
unicode_180987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 20), 'unicode', u'#9e3623')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180986, unicode_180987))
# Adding element type (key, value) (line 43)
unicode_180988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 4), 'unicode', u'dark rose')
unicode_180989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 17), 'unicode', u'#b5485d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180988, unicode_180989))
# Adding element type (key, value) (line 43)
unicode_180990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 4), 'unicode', u'mud')
unicode_180991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 11), 'unicode', u'#735c12')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180990, unicode_180991))
# Adding element type (key, value) (line 43)
unicode_180992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 4), 'unicode', u'brownish')
unicode_180993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 16), 'unicode', u'#9c6d57')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180992, unicode_180993))
# Adding element type (key, value) (line 43)
unicode_180994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 4), 'unicode', u'emerald green')
unicode_180995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 764, 21), 'unicode', u'#028f1e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180994, unicode_180995))
# Adding element type (key, value) (line 43)
unicode_180996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 765, 4), 'unicode', u'pale brown')
unicode_180997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 765, 18), 'unicode', u'#b1916e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180996, unicode_180997))
# Adding element type (key, value) (line 43)
unicode_180998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 766, 4), 'unicode', u'dull blue')
unicode_180999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 766, 17), 'unicode', u'#49759c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_180998, unicode_180999))
# Adding element type (key, value) (line 43)
unicode_181000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 4), 'unicode', u'burnt umber')
unicode_181001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 19), 'unicode', u'#a0450e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181000, unicode_181001))
# Adding element type (key, value) (line 43)
unicode_181002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 4), 'unicode', u'medium green')
unicode_181003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 20), 'unicode', u'#39ad48')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181002, unicode_181003))
# Adding element type (key, value) (line 43)
unicode_181004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 769, 4), 'unicode', u'clay')
unicode_181005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 769, 12), 'unicode', u'#b66a50')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181004, unicode_181005))
# Adding element type (key, value) (line 43)
unicode_181006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 770, 4), 'unicode', u'light aqua')
unicode_181007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 770, 18), 'unicode', u'#8cffdb')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181006, unicode_181007))
# Adding element type (key, value) (line 43)
unicode_181008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 4), 'unicode', u'light olive green')
unicode_181009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 25), 'unicode', u'#a4be5c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181008, unicode_181009))
# Adding element type (key, value) (line 43)
unicode_181010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, 4), 'unicode', u'brownish orange')
unicode_181011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, 23), 'unicode', u'#cb7723')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181010, unicode_181011))
# Adding element type (key, value) (line 43)
unicode_181012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 773, 4), 'unicode', u'dark aqua')
unicode_181013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 773, 17), 'unicode', u'#05696b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181012, unicode_181013))
# Adding element type (key, value) (line 43)
unicode_181014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 774, 4), 'unicode', u'purplish pink')
unicode_181015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 774, 21), 'unicode', u'#ce5dae')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181014, unicode_181015))
# Adding element type (key, value) (line 43)
unicode_181016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, 4), 'unicode', u'dark salmon')
unicode_181017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, 19), 'unicode', u'#c85a53')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181016, unicode_181017))
# Adding element type (key, value) (line 43)
unicode_181018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, 4), 'unicode', u'greenish grey')
unicode_181019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, 21), 'unicode', u'#96ae8d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181018, unicode_181019))
# Adding element type (key, value) (line 43)
unicode_181020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 4), 'unicode', u'jade')
unicode_181021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 12), 'unicode', u'#1fa774')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181020, unicode_181021))
# Adding element type (key, value) (line 43)
unicode_181022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 4), 'unicode', u'ugly green')
unicode_181023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 778, 18), 'unicode', u'#7a9703')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181022, unicode_181023))
# Adding element type (key, value) (line 43)
unicode_181024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, 4), 'unicode', u'dark beige')
unicode_181025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, 18), 'unicode', u'#ac9362')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181024, unicode_181025))
# Adding element type (key, value) (line 43)
unicode_181026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 4), 'unicode', u'emerald')
unicode_181027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 15), 'unicode', u'#01a049')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181026, unicode_181027))
# Adding element type (key, value) (line 43)
unicode_181028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 781, 4), 'unicode', u'pale red')
unicode_181029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 781, 16), 'unicode', u'#d9544d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181028, unicode_181029))
# Adding element type (key, value) (line 43)
unicode_181030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 782, 4), 'unicode', u'light magenta')
unicode_181031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 782, 21), 'unicode', u'#fa5ff7')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181030, unicode_181031))
# Adding element type (key, value) (line 43)
unicode_181032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 4), 'unicode', u'sky')
unicode_181033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 11), 'unicode', u'#82cafc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181032, unicode_181033))
# Adding element type (key, value) (line 43)
unicode_181034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 4), 'unicode', u'light cyan')
unicode_181035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 18), 'unicode', u'#acfffc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181034, unicode_181035))
# Adding element type (key, value) (line 43)
unicode_181036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 4), 'unicode', u'yellow orange')
unicode_181037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 21), 'unicode', u'#fcb001')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181036, unicode_181037))
# Adding element type (key, value) (line 43)
unicode_181038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 786, 4), 'unicode', u'reddish purple')
unicode_181039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 786, 22), 'unicode', u'#910951')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181038, unicode_181039))
# Adding element type (key, value) (line 43)
unicode_181040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 787, 4), 'unicode', u'reddish pink')
unicode_181041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 787, 20), 'unicode', u'#fe2c54')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181040, unicode_181041))
# Adding element type (key, value) (line 43)
unicode_181042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, 4), 'unicode', u'orchid')
unicode_181043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, 14), 'unicode', u'#c875c4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181042, unicode_181043))
# Adding element type (key, value) (line 43)
unicode_181044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 789, 4), 'unicode', u'dirty yellow')
unicode_181045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 789, 20), 'unicode', u'#cdc50a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181044, unicode_181045))
# Adding element type (key, value) (line 43)
unicode_181046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 790, 4), 'unicode', u'orange red')
unicode_181047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 790, 18), 'unicode', u'#fd411e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181046, unicode_181047))
# Adding element type (key, value) (line 43)
unicode_181048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 791, 4), 'unicode', u'deep red')
unicode_181049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 791, 16), 'unicode', u'#9a0200')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181048, unicode_181049))
# Adding element type (key, value) (line 43)
unicode_181050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 792, 4), 'unicode', u'orange brown')
unicode_181051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 792, 20), 'unicode', u'#be6400')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181050, unicode_181051))
# Adding element type (key, value) (line 43)
unicode_181052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 793, 4), 'unicode', u'cobalt blue')
unicode_181053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 793, 19), 'unicode', u'#030aa7')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181052, unicode_181053))
# Adding element type (key, value) (line 43)
unicode_181054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 794, 4), 'unicode', u'neon pink')
unicode_181055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 794, 17), 'unicode', u'#fe019a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181054, unicode_181055))
# Adding element type (key, value) (line 43)
unicode_181056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 795, 4), 'unicode', u'rose pink')
unicode_181057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 795, 17), 'unicode', u'#f7879a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181056, unicode_181057))
# Adding element type (key, value) (line 43)
unicode_181058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 796, 4), 'unicode', u'greyish purple')
unicode_181059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 796, 22), 'unicode', u'#887191')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181058, unicode_181059))
# Adding element type (key, value) (line 43)
unicode_181060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 797, 4), 'unicode', u'raspberry')
unicode_181061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 797, 17), 'unicode', u'#b00149')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181060, unicode_181061))
# Adding element type (key, value) (line 43)
unicode_181062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 798, 4), 'unicode', u'aqua green')
unicode_181063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 798, 18), 'unicode', u'#12e193')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181062, unicode_181063))
# Adding element type (key, value) (line 43)
unicode_181064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 799, 4), 'unicode', u'salmon pink')
unicode_181065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 799, 19), 'unicode', u'#fe7b7c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181064, unicode_181065))
# Adding element type (key, value) (line 43)
unicode_181066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 800, 4), 'unicode', u'tangerine')
unicode_181067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 800, 17), 'unicode', u'#ff9408')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181066, unicode_181067))
# Adding element type (key, value) (line 43)
unicode_181068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 801, 4), 'unicode', u'brownish green')
unicode_181069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 801, 22), 'unicode', u'#6a6e09')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181068, unicode_181069))
# Adding element type (key, value) (line 43)
unicode_181070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 802, 4), 'unicode', u'red brown')
unicode_181071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 802, 17), 'unicode', u'#8b2e16')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181070, unicode_181071))
# Adding element type (key, value) (line 43)
unicode_181072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 803, 4), 'unicode', u'greenish brown')
unicode_181073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 803, 22), 'unicode', u'#696112')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181072, unicode_181073))
# Adding element type (key, value) (line 43)
unicode_181074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 804, 4), 'unicode', u'pumpkin')
unicode_181075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 804, 15), 'unicode', u'#e17701')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181074, unicode_181075))
# Adding element type (key, value) (line 43)
unicode_181076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 805, 4), 'unicode', u'pine green')
unicode_181077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 805, 18), 'unicode', u'#0a481e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181076, unicode_181077))
# Adding element type (key, value) (line 43)
unicode_181078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 806, 4), 'unicode', u'charcoal')
unicode_181079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 806, 16), 'unicode', u'#343837')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181078, unicode_181079))
# Adding element type (key, value) (line 43)
unicode_181080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 4), 'unicode', u'baby pink')
unicode_181081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 17), 'unicode', u'#ffb7ce')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181080, unicode_181081))
# Adding element type (key, value) (line 43)
unicode_181082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 4), 'unicode', u'cornflower')
unicode_181083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 18), 'unicode', u'#6a79f7')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181082, unicode_181083))
# Adding element type (key, value) (line 43)
unicode_181084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 809, 4), 'unicode', u'blue violet')
unicode_181085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 809, 19), 'unicode', u'#5d06e9')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181084, unicode_181085))
# Adding element type (key, value) (line 43)
unicode_181086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 810, 4), 'unicode', u'chocolate')
unicode_181087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 810, 17), 'unicode', u'#3d1c02')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181086, unicode_181087))
# Adding element type (key, value) (line 43)
unicode_181088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 811, 4), 'unicode', u'greyish green')
unicode_181089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 811, 21), 'unicode', u'#82a67d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181088, unicode_181089))
# Adding element type (key, value) (line 43)
unicode_181090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 812, 4), 'unicode', u'scarlet')
unicode_181091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 812, 15), 'unicode', u'#be0119')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181090, unicode_181091))
# Adding element type (key, value) (line 43)
unicode_181092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 4), 'unicode', u'green yellow')
unicode_181093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 20), 'unicode', u'#c9ff27')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181092, unicode_181093))
# Adding element type (key, value) (line 43)
unicode_181094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 814, 4), 'unicode', u'dark olive')
unicode_181095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 814, 18), 'unicode', u'#373e02')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181094, unicode_181095))
# Adding element type (key, value) (line 43)
unicode_181096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 4), 'unicode', u'sienna')
unicode_181097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 14), 'unicode', u'#a9561e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181096, unicode_181097))
# Adding element type (key, value) (line 43)
unicode_181098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 816, 4), 'unicode', u'pastel purple')
unicode_181099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 816, 21), 'unicode', u'#caa0ff')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181098, unicode_181099))
# Adding element type (key, value) (line 43)
unicode_181100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 817, 4), 'unicode', u'terracotta')
unicode_181101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 817, 18), 'unicode', u'#ca6641')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181100, unicode_181101))
# Adding element type (key, value) (line 43)
unicode_181102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 4), 'unicode', u'aqua blue')
unicode_181103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 17), 'unicode', u'#02d8e9')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181102, unicode_181103))
# Adding element type (key, value) (line 43)
unicode_181104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 819, 4), 'unicode', u'sage green')
unicode_181105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 819, 18), 'unicode', u'#88b378')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181104, unicode_181105))
# Adding element type (key, value) (line 43)
unicode_181106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 820, 4), 'unicode', u'blood red')
unicode_181107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 820, 17), 'unicode', u'#980002')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181106, unicode_181107))
# Adding element type (key, value) (line 43)
unicode_181108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 821, 4), 'unicode', u'deep pink')
unicode_181109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 821, 17), 'unicode', u'#cb0162')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181108, unicode_181109))
# Adding element type (key, value) (line 43)
unicode_181110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 822, 4), 'unicode', u'grass')
unicode_181111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 822, 13), 'unicode', u'#5cac2d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181110, unicode_181111))
# Adding element type (key, value) (line 43)
unicode_181112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 823, 4), 'unicode', u'moss')
unicode_181113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 823, 12), 'unicode', u'#769958')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181112, unicode_181113))
# Adding element type (key, value) (line 43)
unicode_181114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 824, 4), 'unicode', u'pastel blue')
unicode_181115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 824, 19), 'unicode', u'#a2bffe')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181114, unicode_181115))
# Adding element type (key, value) (line 43)
unicode_181116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 825, 4), 'unicode', u'bluish green')
unicode_181117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 825, 20), 'unicode', u'#10a674')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181116, unicode_181117))
# Adding element type (key, value) (line 43)
unicode_181118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 826, 4), 'unicode', u'green blue')
unicode_181119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 826, 18), 'unicode', u'#06b48b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181118, unicode_181119))
# Adding element type (key, value) (line 43)
unicode_181120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 827, 4), 'unicode', u'dark tan')
unicode_181121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 827, 16), 'unicode', u'#af884a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181120, unicode_181121))
# Adding element type (key, value) (line 43)
unicode_181122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 828, 4), 'unicode', u'greenish blue')
unicode_181123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 828, 21), 'unicode', u'#0b8b87')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181122, unicode_181123))
# Adding element type (key, value) (line 43)
unicode_181124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 829, 4), 'unicode', u'pale orange')
unicode_181125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 829, 19), 'unicode', u'#ffa756')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181124, unicode_181125))
# Adding element type (key, value) (line 43)
unicode_181126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 830, 4), 'unicode', u'vomit')
unicode_181127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 830, 13), 'unicode', u'#a2a415')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181126, unicode_181127))
# Adding element type (key, value) (line 43)
unicode_181128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 831, 4), 'unicode', u'forrest green')
unicode_181129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 831, 21), 'unicode', u'#154406')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181128, unicode_181129))
# Adding element type (key, value) (line 43)
unicode_181130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 832, 4), 'unicode', u'dark lavender')
unicode_181131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 832, 21), 'unicode', u'#856798')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181130, unicode_181131))
# Adding element type (key, value) (line 43)
unicode_181132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 833, 4), 'unicode', u'dark violet')
unicode_181133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 833, 19), 'unicode', u'#34013f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181132, unicode_181133))
# Adding element type (key, value) (line 43)
unicode_181134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 834, 4), 'unicode', u'purple blue')
unicode_181135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 834, 19), 'unicode', u'#632de9')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181134, unicode_181135))
# Adding element type (key, value) (line 43)
unicode_181136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 835, 4), 'unicode', u'dark cyan')
unicode_181137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 835, 17), 'unicode', u'#0a888a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181136, unicode_181137))
# Adding element type (key, value) (line 43)
unicode_181138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 836, 4), 'unicode', u'olive drab')
unicode_181139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 836, 18), 'unicode', u'#6f7632')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181138, unicode_181139))
# Adding element type (key, value) (line 43)
unicode_181140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 837, 4), 'unicode', u'pinkish')
unicode_181141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 837, 15), 'unicode', u'#d46a7e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181140, unicode_181141))
# Adding element type (key, value) (line 43)
unicode_181142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, 4), 'unicode', u'cobalt')
unicode_181143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, 14), 'unicode', u'#1e488f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181142, unicode_181143))
# Adding element type (key, value) (line 43)
unicode_181144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 839, 4), 'unicode', u'neon purple')
unicode_181145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 839, 19), 'unicode', u'#bc13fe')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181144, unicode_181145))
# Adding element type (key, value) (line 43)
unicode_181146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 840, 4), 'unicode', u'light turquoise')
unicode_181147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 840, 23), 'unicode', u'#7ef4cc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181146, unicode_181147))
# Adding element type (key, value) (line 43)
unicode_181148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 841, 4), 'unicode', u'apple green')
unicode_181149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 841, 19), 'unicode', u'#76cd26')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181148, unicode_181149))
# Adding element type (key, value) (line 43)
unicode_181150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, 4), 'unicode', u'dull green')
unicode_181151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, 18), 'unicode', u'#74a662')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181150, unicode_181151))
# Adding element type (key, value) (line 43)
unicode_181152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 843, 4), 'unicode', u'wine')
unicode_181153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 843, 12), 'unicode', u'#80013f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181152, unicode_181153))
# Adding element type (key, value) (line 43)
unicode_181154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 844, 4), 'unicode', u'powder blue')
unicode_181155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 844, 19), 'unicode', u'#b1d1fc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181154, unicode_181155))
# Adding element type (key, value) (line 43)
unicode_181156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 845, 4), 'unicode', u'off white')
unicode_181157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 845, 17), 'unicode', u'#ffffe4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181156, unicode_181157))
# Adding element type (key, value) (line 43)
unicode_181158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 846, 4), 'unicode', u'electric blue')
unicode_181159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 846, 21), 'unicode', u'#0652ff')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181158, unicode_181159))
# Adding element type (key, value) (line 43)
unicode_181160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 847, 4), 'unicode', u'dark turquoise')
unicode_181161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 847, 22), 'unicode', u'#045c5a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181160, unicode_181161))
# Adding element type (key, value) (line 43)
unicode_181162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 848, 4), 'unicode', u'blue purple')
unicode_181163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 848, 19), 'unicode', u'#5729ce')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181162, unicode_181163))
# Adding element type (key, value) (line 43)
unicode_181164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 849, 4), 'unicode', u'azure')
unicode_181165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 849, 13), 'unicode', u'#069af3')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181164, unicode_181165))
# Adding element type (key, value) (line 43)
unicode_181166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 850, 4), 'unicode', u'bright red')
unicode_181167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 850, 18), 'unicode', u'#ff000d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181166, unicode_181167))
# Adding element type (key, value) (line 43)
unicode_181168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 851, 4), 'unicode', u'pinkish red')
unicode_181169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 851, 19), 'unicode', u'#f10c45')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181168, unicode_181169))
# Adding element type (key, value) (line 43)
unicode_181170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 852, 4), 'unicode', u'cornflower blue')
unicode_181171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 852, 23), 'unicode', u'#5170d7')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181170, unicode_181171))
# Adding element type (key, value) (line 43)
unicode_181172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 853, 4), 'unicode', u'light olive')
unicode_181173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 853, 19), 'unicode', u'#acbf69')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181172, unicode_181173))
# Adding element type (key, value) (line 43)
unicode_181174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 854, 4), 'unicode', u'grape')
unicode_181175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 854, 13), 'unicode', u'#6c3461')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181174, unicode_181175))
# Adding element type (key, value) (line 43)
unicode_181176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 855, 4), 'unicode', u'greyish blue')
unicode_181177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 855, 20), 'unicode', u'#5e819d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181176, unicode_181177))
# Adding element type (key, value) (line 43)
unicode_181178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 856, 4), 'unicode', u'purplish blue')
unicode_181179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 856, 21), 'unicode', u'#601ef9')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181178, unicode_181179))
# Adding element type (key, value) (line 43)
unicode_181180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 857, 4), 'unicode', u'yellowish green')
unicode_181181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 857, 23), 'unicode', u'#b0dd16')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181180, unicode_181181))
# Adding element type (key, value) (line 43)
unicode_181182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 858, 4), 'unicode', u'greenish yellow')
unicode_181183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 858, 23), 'unicode', u'#cdfd02')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181182, unicode_181183))
# Adding element type (key, value) (line 43)
unicode_181184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 859, 4), 'unicode', u'medium blue')
unicode_181185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 859, 19), 'unicode', u'#2c6fbb')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181184, unicode_181185))
# Adding element type (key, value) (line 43)
unicode_181186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 860, 4), 'unicode', u'dusty rose')
unicode_181187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 860, 18), 'unicode', u'#c0737a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181186, unicode_181187))
# Adding element type (key, value) (line 43)
unicode_181188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 861, 4), 'unicode', u'light violet')
unicode_181189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 861, 20), 'unicode', u'#d6b4fc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181188, unicode_181189))
# Adding element type (key, value) (line 43)
unicode_181190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 862, 4), 'unicode', u'midnight blue')
unicode_181191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 862, 21), 'unicode', u'#020035')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181190, unicode_181191))
# Adding element type (key, value) (line 43)
unicode_181192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 863, 4), 'unicode', u'bluish purple')
unicode_181193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 863, 21), 'unicode', u'#703be7')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181192, unicode_181193))
# Adding element type (key, value) (line 43)
unicode_181194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 864, 4), 'unicode', u'red orange')
unicode_181195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 864, 18), 'unicode', u'#fd3c06')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181194, unicode_181195))
# Adding element type (key, value) (line 43)
unicode_181196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 865, 4), 'unicode', u'dark magenta')
unicode_181197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 865, 20), 'unicode', u'#960056')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181196, unicode_181197))
# Adding element type (key, value) (line 43)
unicode_181198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 866, 4), 'unicode', u'greenish')
unicode_181199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 866, 16), 'unicode', u'#40a368')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181198, unicode_181199))
# Adding element type (key, value) (line 43)
unicode_181200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 867, 4), 'unicode', u'ocean blue')
unicode_181201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 867, 18), 'unicode', u'#03719c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181200, unicode_181201))
# Adding element type (key, value) (line 43)
unicode_181202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 868, 4), 'unicode', u'coral')
unicode_181203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 868, 13), 'unicode', u'#fc5a50')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181202, unicode_181203))
# Adding element type (key, value) (line 43)
unicode_181204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 869, 4), 'unicode', u'cream')
unicode_181205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 869, 13), 'unicode', u'#ffffc2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181204, unicode_181205))
# Adding element type (key, value) (line 43)
unicode_181206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 870, 4), 'unicode', u'reddish brown')
unicode_181207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 870, 21), 'unicode', u'#7f2b0a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181206, unicode_181207))
# Adding element type (key, value) (line 43)
unicode_181208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 871, 4), 'unicode', u'burnt sienna')
unicode_181209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 871, 20), 'unicode', u'#b04e0f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181208, unicode_181209))
# Adding element type (key, value) (line 43)
unicode_181210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 872, 4), 'unicode', u'brick')
unicode_181211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 872, 13), 'unicode', u'#a03623')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181210, unicode_181211))
# Adding element type (key, value) (line 43)
unicode_181212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 873, 4), 'unicode', u'sage')
unicode_181213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 873, 12), 'unicode', u'#87ae73')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181212, unicode_181213))
# Adding element type (key, value) (line 43)
unicode_181214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 874, 4), 'unicode', u'grey green')
unicode_181215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 874, 18), 'unicode', u'#789b73')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181214, unicode_181215))
# Adding element type (key, value) (line 43)
unicode_181216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 875, 4), 'unicode', u'white')
unicode_181217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 875, 13), 'unicode', u'#ffffff')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181216, unicode_181217))
# Adding element type (key, value) (line 43)
unicode_181218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 876, 4), 'unicode', u"robin's egg blue")
unicode_181219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 876, 24), 'unicode', u'#98eff9')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181218, unicode_181219))
# Adding element type (key, value) (line 43)
unicode_181220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 877, 4), 'unicode', u'moss green')
unicode_181221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 877, 18), 'unicode', u'#658b38')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181220, unicode_181221))
# Adding element type (key, value) (line 43)
unicode_181222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 878, 4), 'unicode', u'steel blue')
unicode_181223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 878, 18), 'unicode', u'#5a7d9a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181222, unicode_181223))
# Adding element type (key, value) (line 43)
unicode_181224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 879, 4), 'unicode', u'eggplant')
unicode_181225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 879, 16), 'unicode', u'#380835')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181224, unicode_181225))
# Adding element type (key, value) (line 43)
unicode_181226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 880, 4), 'unicode', u'light yellow')
unicode_181227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 880, 20), 'unicode', u'#fffe7a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181226, unicode_181227))
# Adding element type (key, value) (line 43)
unicode_181228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 881, 4), 'unicode', u'leaf green')
unicode_181229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 881, 18), 'unicode', u'#5ca904')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181228, unicode_181229))
# Adding element type (key, value) (line 43)
unicode_181230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 882, 4), 'unicode', u'light grey')
unicode_181231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 882, 18), 'unicode', u'#d8dcd6')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181230, unicode_181231))
# Adding element type (key, value) (line 43)
unicode_181232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 883, 4), 'unicode', u'puke')
unicode_181233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 883, 12), 'unicode', u'#a5a502')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181232, unicode_181233))
# Adding element type (key, value) (line 43)
unicode_181234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 884, 4), 'unicode', u'pinkish purple')
unicode_181235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 884, 22), 'unicode', u'#d648d7')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181234, unicode_181235))
# Adding element type (key, value) (line 43)
unicode_181236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 885, 4), 'unicode', u'sea blue')
unicode_181237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 885, 16), 'unicode', u'#047495')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181236, unicode_181237))
# Adding element type (key, value) (line 43)
unicode_181238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 886, 4), 'unicode', u'pale purple')
unicode_181239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 886, 19), 'unicode', u'#b790d4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181238, unicode_181239))
# Adding element type (key, value) (line 43)
unicode_181240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 887, 4), 'unicode', u'slate blue')
unicode_181241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 887, 18), 'unicode', u'#5b7c99')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181240, unicode_181241))
# Adding element type (key, value) (line 43)
unicode_181242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 888, 4), 'unicode', u'blue grey')
unicode_181243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 888, 17), 'unicode', u'#607c8e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181242, unicode_181243))
# Adding element type (key, value) (line 43)
unicode_181244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 889, 4), 'unicode', u'hunter green')
unicode_181245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 889, 20), 'unicode', u'#0b4008')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181244, unicode_181245))
# Adding element type (key, value) (line 43)
unicode_181246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 890, 4), 'unicode', u'fuchsia')
unicode_181247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 890, 15), 'unicode', u'#ed0dd9')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181246, unicode_181247))
# Adding element type (key, value) (line 43)
unicode_181248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 891, 4), 'unicode', u'crimson')
unicode_181249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 891, 15), 'unicode', u'#8c000f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181248, unicode_181249))
# Adding element type (key, value) (line 43)
unicode_181250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 892, 4), 'unicode', u'pale yellow')
unicode_181251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 892, 19), 'unicode', u'#ffff84')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181250, unicode_181251))
# Adding element type (key, value) (line 43)
unicode_181252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 893, 4), 'unicode', u'ochre')
unicode_181253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 893, 13), 'unicode', u'#bf9005')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181252, unicode_181253))
# Adding element type (key, value) (line 43)
unicode_181254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 894, 4), 'unicode', u'mustard yellow')
unicode_181255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 894, 22), 'unicode', u'#d2bd0a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181254, unicode_181255))
# Adding element type (key, value) (line 43)
unicode_181256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 895, 4), 'unicode', u'light red')
unicode_181257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 895, 17), 'unicode', u'#ff474c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181256, unicode_181257))
# Adding element type (key, value) (line 43)
unicode_181258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 896, 4), 'unicode', u'cerulean')
unicode_181259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 896, 16), 'unicode', u'#0485d1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181258, unicode_181259))
# Adding element type (key, value) (line 43)
unicode_181260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 897, 4), 'unicode', u'pale pink')
unicode_181261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 897, 17), 'unicode', u'#ffcfdc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181260, unicode_181261))
# Adding element type (key, value) (line 43)
unicode_181262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 898, 4), 'unicode', u'deep blue')
unicode_181263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 898, 17), 'unicode', u'#040273')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181262, unicode_181263))
# Adding element type (key, value) (line 43)
unicode_181264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 899, 4), 'unicode', u'rust')
unicode_181265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 899, 12), 'unicode', u'#a83c09')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181264, unicode_181265))
# Adding element type (key, value) (line 43)
unicode_181266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 900, 4), 'unicode', u'light teal')
unicode_181267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 900, 18), 'unicode', u'#90e4c1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181266, unicode_181267))
# Adding element type (key, value) (line 43)
unicode_181268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 901, 4), 'unicode', u'slate')
unicode_181269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 901, 13), 'unicode', u'#516572')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181268, unicode_181269))
# Adding element type (key, value) (line 43)
unicode_181270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 902, 4), 'unicode', u'goldenrod')
unicode_181271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 902, 17), 'unicode', u'#fac205')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181270, unicode_181271))
# Adding element type (key, value) (line 43)
unicode_181272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 903, 4), 'unicode', u'dark yellow')
unicode_181273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 903, 19), 'unicode', u'#d5b60a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181272, unicode_181273))
# Adding element type (key, value) (line 43)
unicode_181274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 904, 4), 'unicode', u'dark grey')
unicode_181275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 904, 17), 'unicode', u'#363737')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181274, unicode_181275))
# Adding element type (key, value) (line 43)
unicode_181276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 905, 4), 'unicode', u'army green')
unicode_181277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 905, 18), 'unicode', u'#4b5d16')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181276, unicode_181277))
# Adding element type (key, value) (line 43)
unicode_181278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 906, 4), 'unicode', u'grey blue')
unicode_181279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 906, 17), 'unicode', u'#6b8ba4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181278, unicode_181279))
# Adding element type (key, value) (line 43)
unicode_181280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 907, 4), 'unicode', u'seafoam')
unicode_181281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 907, 15), 'unicode', u'#80f9ad')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181280, unicode_181281))
# Adding element type (key, value) (line 43)
unicode_181282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 908, 4), 'unicode', u'puce')
unicode_181283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 908, 12), 'unicode', u'#a57e52')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181282, unicode_181283))
# Adding element type (key, value) (line 43)
unicode_181284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 909, 4), 'unicode', u'spring green')
unicode_181285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 909, 20), 'unicode', u'#a9f971')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181284, unicode_181285))
# Adding element type (key, value) (line 43)
unicode_181286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 910, 4), 'unicode', u'dark orange')
unicode_181287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 910, 19), 'unicode', u'#c65102')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181286, unicode_181287))
# Adding element type (key, value) (line 43)
unicode_181288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 911, 4), 'unicode', u'sand')
unicode_181289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 911, 12), 'unicode', u'#e2ca76')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181288, unicode_181289))
# Adding element type (key, value) (line 43)
unicode_181290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 912, 4), 'unicode', u'pastel green')
unicode_181291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 912, 20), 'unicode', u'#b0ff9d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181290, unicode_181291))
# Adding element type (key, value) (line 43)
unicode_181292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 913, 4), 'unicode', u'mint')
unicode_181293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 913, 12), 'unicode', u'#9ffeb0')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181292, unicode_181293))
# Adding element type (key, value) (line 43)
unicode_181294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 914, 4), 'unicode', u'light orange')
unicode_181295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 914, 20), 'unicode', u'#fdaa48')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181294, unicode_181295))
# Adding element type (key, value) (line 43)
unicode_181296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 915, 4), 'unicode', u'bright pink')
unicode_181297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 915, 19), 'unicode', u'#fe01b1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181296, unicode_181297))
# Adding element type (key, value) (line 43)
unicode_181298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 916, 4), 'unicode', u'chartreuse')
unicode_181299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 916, 18), 'unicode', u'#c1f80a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181298, unicode_181299))
# Adding element type (key, value) (line 43)
unicode_181300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 917, 4), 'unicode', u'deep purple')
unicode_181301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 917, 19), 'unicode', u'#36013f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181300, unicode_181301))
# Adding element type (key, value) (line 43)
unicode_181302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 918, 4), 'unicode', u'dark brown')
unicode_181303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 918, 18), 'unicode', u'#341c02')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181302, unicode_181303))
# Adding element type (key, value) (line 43)
unicode_181304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 919, 4), 'unicode', u'taupe')
unicode_181305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 919, 13), 'unicode', u'#b9a281')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181304, unicode_181305))
# Adding element type (key, value) (line 43)
unicode_181306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 920, 4), 'unicode', u'pea green')
unicode_181307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 920, 17), 'unicode', u'#8eab12')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181306, unicode_181307))
# Adding element type (key, value) (line 43)
unicode_181308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 921, 4), 'unicode', u'puke green')
unicode_181309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 921, 18), 'unicode', u'#9aae07')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181308, unicode_181309))
# Adding element type (key, value) (line 43)
unicode_181310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 4), 'unicode', u'kelly green')
unicode_181311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 19), 'unicode', u'#02ab2e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181310, unicode_181311))
# Adding element type (key, value) (line 43)
unicode_181312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 923, 4), 'unicode', u'seafoam green')
unicode_181313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 923, 21), 'unicode', u'#7af9ab')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181312, unicode_181313))
# Adding element type (key, value) (line 43)
unicode_181314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 924, 4), 'unicode', u'blue green')
unicode_181315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 924, 18), 'unicode', u'#137e6d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181314, unicode_181315))
# Adding element type (key, value) (line 43)
unicode_181316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 925, 4), 'unicode', u'khaki')
unicode_181317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 925, 13), 'unicode', u'#aaa662')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181316, unicode_181317))
# Adding element type (key, value) (line 43)
unicode_181318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 926, 4), 'unicode', u'burgundy')
unicode_181319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 926, 16), 'unicode', u'#610023')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181318, unicode_181319))
# Adding element type (key, value) (line 43)
unicode_181320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 927, 4), 'unicode', u'dark teal')
unicode_181321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 927, 17), 'unicode', u'#014d4e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181320, unicode_181321))
# Adding element type (key, value) (line 43)
unicode_181322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 928, 4), 'unicode', u'brick red')
unicode_181323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 928, 17), 'unicode', u'#8f1402')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181322, unicode_181323))
# Adding element type (key, value) (line 43)
unicode_181324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 929, 4), 'unicode', u'royal purple')
unicode_181325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 929, 20), 'unicode', u'#4b006e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181324, unicode_181325))
# Adding element type (key, value) (line 43)
unicode_181326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 930, 4), 'unicode', u'plum')
unicode_181327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 930, 12), 'unicode', u'#580f41')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181326, unicode_181327))
# Adding element type (key, value) (line 43)
unicode_181328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 931, 4), 'unicode', u'mint green')
unicode_181329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 931, 18), 'unicode', u'#8fff9f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181328, unicode_181329))
# Adding element type (key, value) (line 43)
unicode_181330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 932, 4), 'unicode', u'gold')
unicode_181331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 932, 12), 'unicode', u'#dbb40c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181330, unicode_181331))
# Adding element type (key, value) (line 43)
unicode_181332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 933, 4), 'unicode', u'baby blue')
unicode_181333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 933, 17), 'unicode', u'#a2cffe')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181332, unicode_181333))
# Adding element type (key, value) (line 43)
unicode_181334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 934, 4), 'unicode', u'yellow green')
unicode_181335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 934, 20), 'unicode', u'#c0fb2d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181334, unicode_181335))
# Adding element type (key, value) (line 43)
unicode_181336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 935, 4), 'unicode', u'bright purple')
unicode_181337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 935, 21), 'unicode', u'#be03fd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181336, unicode_181337))
# Adding element type (key, value) (line 43)
unicode_181338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 936, 4), 'unicode', u'dark red')
unicode_181339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 936, 16), 'unicode', u'#840000')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181338, unicode_181339))
# Adding element type (key, value) (line 43)
unicode_181340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 937, 4), 'unicode', u'pale blue')
unicode_181341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 937, 17), 'unicode', u'#d0fefe')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181340, unicode_181341))
# Adding element type (key, value) (line 43)
unicode_181342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 938, 4), 'unicode', u'grass green')
unicode_181343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 938, 19), 'unicode', u'#3f9b0b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181342, unicode_181343))
# Adding element type (key, value) (line 43)
unicode_181344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 939, 4), 'unicode', u'navy')
unicode_181345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 939, 12), 'unicode', u'#01153e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181344, unicode_181345))
# Adding element type (key, value) (line 43)
unicode_181346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 940, 4), 'unicode', u'aquamarine')
unicode_181347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 940, 18), 'unicode', u'#04d8b2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181346, unicode_181347))
# Adding element type (key, value) (line 43)
unicode_181348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 941, 4), 'unicode', u'burnt orange')
unicode_181349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 941, 20), 'unicode', u'#c04e01')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181348, unicode_181349))
# Adding element type (key, value) (line 43)
unicode_181350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 942, 4), 'unicode', u'neon green')
unicode_181351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 942, 18), 'unicode', u'#0cff0c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181350, unicode_181351))
# Adding element type (key, value) (line 43)
unicode_181352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 943, 4), 'unicode', u'bright blue')
unicode_181353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 943, 19), 'unicode', u'#0165fc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181352, unicode_181353))
# Adding element type (key, value) (line 43)
unicode_181354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 944, 4), 'unicode', u'rose')
unicode_181355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 944, 12), 'unicode', u'#cf6275')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181354, unicode_181355))
# Adding element type (key, value) (line 43)
unicode_181356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 945, 4), 'unicode', u'light pink')
unicode_181357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 945, 18), 'unicode', u'#ffd1df')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181356, unicode_181357))
# Adding element type (key, value) (line 43)
unicode_181358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 946, 4), 'unicode', u'mustard')
unicode_181359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 946, 15), 'unicode', u'#ceb301')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181358, unicode_181359))
# Adding element type (key, value) (line 43)
unicode_181360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 947, 4), 'unicode', u'indigo')
unicode_181361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 947, 14), 'unicode', u'#380282')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181360, unicode_181361))
# Adding element type (key, value) (line 43)
unicode_181362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 948, 4), 'unicode', u'lime')
unicode_181363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 948, 12), 'unicode', u'#aaff32')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181362, unicode_181363))
# Adding element type (key, value) (line 43)
unicode_181364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 949, 4), 'unicode', u'sea green')
unicode_181365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 949, 17), 'unicode', u'#53fca1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181364, unicode_181365))
# Adding element type (key, value) (line 43)
unicode_181366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 950, 4), 'unicode', u'periwinkle')
unicode_181367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 950, 18), 'unicode', u'#8e82fe')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181366, unicode_181367))
# Adding element type (key, value) (line 43)
unicode_181368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 951, 4), 'unicode', u'dark pink')
unicode_181369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 951, 17), 'unicode', u'#cb416b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181368, unicode_181369))
# Adding element type (key, value) (line 43)
unicode_181370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 952, 4), 'unicode', u'olive green')
unicode_181371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 952, 19), 'unicode', u'#677a04')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181370, unicode_181371))
# Adding element type (key, value) (line 43)
unicode_181372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 953, 4), 'unicode', u'peach')
unicode_181373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 953, 13), 'unicode', u'#ffb07c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181372, unicode_181373))
# Adding element type (key, value) (line 43)
unicode_181374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 954, 4), 'unicode', u'pale green')
unicode_181375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 954, 18), 'unicode', u'#c7fdb5')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181374, unicode_181375))
# Adding element type (key, value) (line 43)
unicode_181376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 955, 4), 'unicode', u'light brown')
unicode_181377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 955, 19), 'unicode', u'#ad8150')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181376, unicode_181377))
# Adding element type (key, value) (line 43)
unicode_181378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 956, 4), 'unicode', u'hot pink')
unicode_181379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 956, 16), 'unicode', u'#ff028d')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181378, unicode_181379))
# Adding element type (key, value) (line 43)
unicode_181380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 957, 4), 'unicode', u'black')
unicode_181381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 957, 13), 'unicode', u'#000000')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181380, unicode_181381))
# Adding element type (key, value) (line 43)
unicode_181382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 958, 4), 'unicode', u'lilac')
unicode_181383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 958, 13), 'unicode', u'#cea2fd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181382, unicode_181383))
# Adding element type (key, value) (line 43)
unicode_181384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 959, 4), 'unicode', u'navy blue')
unicode_181385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 959, 17), 'unicode', u'#001146')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181384, unicode_181385))
# Adding element type (key, value) (line 43)
unicode_181386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 4), 'unicode', u'royal blue')
unicode_181387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 18), 'unicode', u'#0504aa')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181386, unicode_181387))
# Adding element type (key, value) (line 43)
unicode_181388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 961, 4), 'unicode', u'beige')
unicode_181389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 961, 13), 'unicode', u'#e6daa6')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181388, unicode_181389))
# Adding element type (key, value) (line 43)
unicode_181390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 962, 4), 'unicode', u'salmon')
unicode_181391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 962, 14), 'unicode', u'#ff796c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181390, unicode_181391))
# Adding element type (key, value) (line 43)
unicode_181392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 963, 4), 'unicode', u'olive')
unicode_181393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 963, 13), 'unicode', u'#6e750e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181392, unicode_181393))
# Adding element type (key, value) (line 43)
unicode_181394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 964, 4), 'unicode', u'maroon')
unicode_181395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 964, 14), 'unicode', u'#650021')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181394, unicode_181395))
# Adding element type (key, value) (line 43)
unicode_181396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 965, 4), 'unicode', u'bright green')
unicode_181397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 965, 20), 'unicode', u'#01ff07')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181396, unicode_181397))
# Adding element type (key, value) (line 43)
unicode_181398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 966, 4), 'unicode', u'dark purple')
unicode_181399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 966, 19), 'unicode', u'#35063e')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181398, unicode_181399))
# Adding element type (key, value) (line 43)
unicode_181400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 967, 4), 'unicode', u'mauve')
unicode_181401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 967, 13), 'unicode', u'#ae7181')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181400, unicode_181401))
# Adding element type (key, value) (line 43)
unicode_181402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 968, 4), 'unicode', u'forest green')
unicode_181403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 968, 20), 'unicode', u'#06470c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181402, unicode_181403))
# Adding element type (key, value) (line 43)
unicode_181404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 969, 4), 'unicode', u'aqua')
unicode_181405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 969, 12), 'unicode', u'#13eac9')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181404, unicode_181405))
# Adding element type (key, value) (line 43)
unicode_181406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 970, 4), 'unicode', u'cyan')
unicode_181407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 970, 12), 'unicode', u'#00ffff')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181406, unicode_181407))
# Adding element type (key, value) (line 43)
unicode_181408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 971, 4), 'unicode', u'tan')
unicode_181409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 971, 11), 'unicode', u'#d1b26f')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181408, unicode_181409))
# Adding element type (key, value) (line 43)
unicode_181410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 972, 4), 'unicode', u'dark blue')
unicode_181411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 972, 17), 'unicode', u'#00035b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181410, unicode_181411))
# Adding element type (key, value) (line 43)
unicode_181412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 973, 4), 'unicode', u'lavender')
unicode_181413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 973, 16), 'unicode', u'#c79fef')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181412, unicode_181413))
# Adding element type (key, value) (line 43)
unicode_181414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 974, 4), 'unicode', u'turquoise')
unicode_181415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 974, 17), 'unicode', u'#06c2ac')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181414, unicode_181415))
# Adding element type (key, value) (line 43)
unicode_181416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 975, 4), 'unicode', u'dark green')
unicode_181417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 975, 18), 'unicode', u'#033500')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181416, unicode_181417))
# Adding element type (key, value) (line 43)
unicode_181418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 976, 4), 'unicode', u'violet')
unicode_181419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 976, 14), 'unicode', u'#9a0eea')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181418, unicode_181419))
# Adding element type (key, value) (line 43)
unicode_181420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 977, 4), 'unicode', u'light purple')
unicode_181421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 977, 20), 'unicode', u'#bf77f6')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181420, unicode_181421))
# Adding element type (key, value) (line 43)
unicode_181422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 978, 4), 'unicode', u'lime green')
unicode_181423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 978, 18), 'unicode', u'#89fe05')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181422, unicode_181423))
# Adding element type (key, value) (line 43)
unicode_181424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 979, 4), 'unicode', u'grey')
unicode_181425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 979, 12), 'unicode', u'#929591')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181424, unicode_181425))
# Adding element type (key, value) (line 43)
unicode_181426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 980, 4), 'unicode', u'sky blue')
unicode_181427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 980, 16), 'unicode', u'#75bbfd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181426, unicode_181427))
# Adding element type (key, value) (line 43)
unicode_181428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 981, 4), 'unicode', u'yellow')
unicode_181429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 981, 14), 'unicode', u'#ffff14')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181428, unicode_181429))
# Adding element type (key, value) (line 43)
unicode_181430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 982, 4), 'unicode', u'magenta')
unicode_181431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 982, 15), 'unicode', u'#c20078')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181430, unicode_181431))
# Adding element type (key, value) (line 43)
unicode_181432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 983, 4), 'unicode', u'light green')
unicode_181433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 983, 19), 'unicode', u'#96f97b')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181432, unicode_181433))
# Adding element type (key, value) (line 43)
unicode_181434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 984, 4), 'unicode', u'orange')
unicode_181435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 984, 14), 'unicode', u'#f97306')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181434, unicode_181435))
# Adding element type (key, value) (line 43)
unicode_181436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 985, 4), 'unicode', u'teal')
unicode_181437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 985, 12), 'unicode', u'#029386')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181436, unicode_181437))
# Adding element type (key, value) (line 43)
unicode_181438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 986, 4), 'unicode', u'light blue')
unicode_181439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 986, 18), 'unicode', u'#95d0fc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181438, unicode_181439))
# Adding element type (key, value) (line 43)
unicode_181440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 987, 4), 'unicode', u'red')
unicode_181441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 987, 11), 'unicode', u'#e50000')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181440, unicode_181441))
# Adding element type (key, value) (line 43)
unicode_181442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 988, 4), 'unicode', u'brown')
unicode_181443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 988, 13), 'unicode', u'#653700')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181442, unicode_181443))
# Adding element type (key, value) (line 43)
unicode_181444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 989, 4), 'unicode', u'pink')
unicode_181445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 989, 12), 'unicode', u'#ff81c0')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181444, unicode_181445))
# Adding element type (key, value) (line 43)
unicode_181446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 990, 4), 'unicode', u'blue')
unicode_181447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 990, 12), 'unicode', u'#0343df')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181446, unicode_181447))
# Adding element type (key, value) (line 43)
unicode_181448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 991, 4), 'unicode', u'green')
unicode_181449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 991, 13), 'unicode', u'#15b01a')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181448, unicode_181449))
# Adding element type (key, value) (line 43)
unicode_181450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 992, 4), 'unicode', u'purple')
unicode_181451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 992, 14), 'unicode', u'#7e1e9c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 14), dict_179553, (unicode_181450, unicode_181451))

# Assigning a type to the variable 'XKCD_COLORS' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'XKCD_COLORS', dict_179553)

# Assigning a DictComp to a Name (line 995):
# Calculating dict comprehension
module_type_store = module_type_store.open_function_context('dict comprehension expression', 995, 15, True)
# Calculating comprehension expression

# Call to items(...): (line 995)
# Processing the call keyword arguments (line 995)
kwargs_181458 = {}
# Getting the type of 'XKCD_COLORS' (line 995)
XKCD_COLORS_181456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 56), 'XKCD_COLORS', False)
# Obtaining the member 'items' of a type (line 995)
items_181457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 995, 56), XKCD_COLORS_181456, 'items')
# Calling items(args, kwargs) (line 995)
items_call_result_181459 = invoke(stypy.reporting.localization.Localization(__file__, 995, 56), items_181457, *[], **kwargs_181458)

comprehension_181460 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 995, 15), items_call_result_181459)
# Assigning a type to the variable 'name' (line 995)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 995, 15), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 995, 15), comprehension_181460))
# Assigning a type to the variable 'value' (line 995)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 995, 15), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 995, 15), comprehension_181460))
unicode_181452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 995, 15), 'unicode', u'xkcd:')
# Getting the type of 'name' (line 995)
name_181453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 25), 'name')
# Applying the binary operator '+' (line 995)
result_add_181454 = python_operator(stypy.reporting.localization.Localization(__file__, 995, 15), '+', unicode_181452, name_181453)

# Getting the type of 'value' (line 995)
value_181455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 31), 'value')
dict_181461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 995, 15), 'dict')
# Destroy the current context
module_type_store = module_type_store.close_function_context()
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 995, 15), dict_181461, (result_add_181454, value_181455))
# Assigning a type to the variable 'XKCD_COLORS' (line 995)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 995, 0), 'XKCD_COLORS', dict_181461)

# Assigning a Dict to a Name (line 999):

# Obtaining an instance of the builtin type 'dict' (line 999)
dict_181462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 999, 14), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 999)
# Adding element type (key, value) (line 999)
unicode_181463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1000, 4), 'unicode', u'aliceblue')
unicode_181464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1000, 28), 'unicode', u'#F0F8FF')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181463, unicode_181464))
# Adding element type (key, value) (line 999)
unicode_181465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1001, 4), 'unicode', u'antiquewhite')
unicode_181466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1001, 28), 'unicode', u'#FAEBD7')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181465, unicode_181466))
# Adding element type (key, value) (line 999)
unicode_181467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1002, 4), 'unicode', u'aqua')
unicode_181468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1002, 28), 'unicode', u'#00FFFF')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181467, unicode_181468))
# Adding element type (key, value) (line 999)
unicode_181469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1003, 4), 'unicode', u'aquamarine')
unicode_181470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1003, 28), 'unicode', u'#7FFFD4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181469, unicode_181470))
# Adding element type (key, value) (line 999)
unicode_181471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1004, 4), 'unicode', u'azure')
unicode_181472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1004, 28), 'unicode', u'#F0FFFF')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181471, unicode_181472))
# Adding element type (key, value) (line 999)
unicode_181473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1005, 4), 'unicode', u'beige')
unicode_181474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1005, 28), 'unicode', u'#F5F5DC')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181473, unicode_181474))
# Adding element type (key, value) (line 999)
unicode_181475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1006, 4), 'unicode', u'bisque')
unicode_181476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1006, 28), 'unicode', u'#FFE4C4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181475, unicode_181476))
# Adding element type (key, value) (line 999)
unicode_181477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1007, 4), 'unicode', u'black')
unicode_181478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1007, 28), 'unicode', u'#000000')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181477, unicode_181478))
# Adding element type (key, value) (line 999)
unicode_181479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1008, 4), 'unicode', u'blanchedalmond')
unicode_181480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1008, 28), 'unicode', u'#FFEBCD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181479, unicode_181480))
# Adding element type (key, value) (line 999)
unicode_181481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1009, 4), 'unicode', u'blue')
unicode_181482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1009, 28), 'unicode', u'#0000FF')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181481, unicode_181482))
# Adding element type (key, value) (line 999)
unicode_181483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1010, 4), 'unicode', u'blueviolet')
unicode_181484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1010, 28), 'unicode', u'#8A2BE2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181483, unicode_181484))
# Adding element type (key, value) (line 999)
unicode_181485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1011, 4), 'unicode', u'brown')
unicode_181486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1011, 28), 'unicode', u'#A52A2A')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181485, unicode_181486))
# Adding element type (key, value) (line 999)
unicode_181487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1012, 4), 'unicode', u'burlywood')
unicode_181488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1012, 28), 'unicode', u'#DEB887')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181487, unicode_181488))
# Adding element type (key, value) (line 999)
unicode_181489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1013, 4), 'unicode', u'cadetblue')
unicode_181490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1013, 28), 'unicode', u'#5F9EA0')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181489, unicode_181490))
# Adding element type (key, value) (line 999)
unicode_181491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1014, 4), 'unicode', u'chartreuse')
unicode_181492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1014, 28), 'unicode', u'#7FFF00')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181491, unicode_181492))
# Adding element type (key, value) (line 999)
unicode_181493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1015, 4), 'unicode', u'chocolate')
unicode_181494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1015, 28), 'unicode', u'#D2691E')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181493, unicode_181494))
# Adding element type (key, value) (line 999)
unicode_181495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1016, 4), 'unicode', u'coral')
unicode_181496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1016, 28), 'unicode', u'#FF7F50')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181495, unicode_181496))
# Adding element type (key, value) (line 999)
unicode_181497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1017, 4), 'unicode', u'cornflowerblue')
unicode_181498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1017, 28), 'unicode', u'#6495ED')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181497, unicode_181498))
# Adding element type (key, value) (line 999)
unicode_181499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1018, 4), 'unicode', u'cornsilk')
unicode_181500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1018, 28), 'unicode', u'#FFF8DC')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181499, unicode_181500))
# Adding element type (key, value) (line 999)
unicode_181501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1019, 4), 'unicode', u'crimson')
unicode_181502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1019, 28), 'unicode', u'#DC143C')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181501, unicode_181502))
# Adding element type (key, value) (line 999)
unicode_181503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1020, 4), 'unicode', u'cyan')
unicode_181504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1020, 28), 'unicode', u'#00FFFF')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181503, unicode_181504))
# Adding element type (key, value) (line 999)
unicode_181505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1021, 4), 'unicode', u'darkblue')
unicode_181506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1021, 28), 'unicode', u'#00008B')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181505, unicode_181506))
# Adding element type (key, value) (line 999)
unicode_181507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1022, 4), 'unicode', u'darkcyan')
unicode_181508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1022, 28), 'unicode', u'#008B8B')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181507, unicode_181508))
# Adding element type (key, value) (line 999)
unicode_181509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1023, 4), 'unicode', u'darkgoldenrod')
unicode_181510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1023, 28), 'unicode', u'#B8860B')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181509, unicode_181510))
# Adding element type (key, value) (line 999)
unicode_181511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1024, 4), 'unicode', u'darkgray')
unicode_181512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1024, 28), 'unicode', u'#A9A9A9')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181511, unicode_181512))
# Adding element type (key, value) (line 999)
unicode_181513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1025, 4), 'unicode', u'darkgreen')
unicode_181514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1025, 28), 'unicode', u'#006400')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181513, unicode_181514))
# Adding element type (key, value) (line 999)
unicode_181515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1026, 4), 'unicode', u'darkgrey')
unicode_181516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1026, 28), 'unicode', u'#A9A9A9')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181515, unicode_181516))
# Adding element type (key, value) (line 999)
unicode_181517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1027, 4), 'unicode', u'darkkhaki')
unicode_181518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1027, 28), 'unicode', u'#BDB76B')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181517, unicode_181518))
# Adding element type (key, value) (line 999)
unicode_181519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1028, 4), 'unicode', u'darkmagenta')
unicode_181520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1028, 28), 'unicode', u'#8B008B')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181519, unicode_181520))
# Adding element type (key, value) (line 999)
unicode_181521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1029, 4), 'unicode', u'darkolivegreen')
unicode_181522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1029, 28), 'unicode', u'#556B2F')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181521, unicode_181522))
# Adding element type (key, value) (line 999)
unicode_181523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1030, 4), 'unicode', u'darkorange')
unicode_181524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1030, 28), 'unicode', u'#FF8C00')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181523, unicode_181524))
# Adding element type (key, value) (line 999)
unicode_181525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1031, 4), 'unicode', u'darkorchid')
unicode_181526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1031, 28), 'unicode', u'#9932CC')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181525, unicode_181526))
# Adding element type (key, value) (line 999)
unicode_181527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1032, 4), 'unicode', u'darkred')
unicode_181528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1032, 28), 'unicode', u'#8B0000')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181527, unicode_181528))
# Adding element type (key, value) (line 999)
unicode_181529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1033, 4), 'unicode', u'darksalmon')
unicode_181530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1033, 28), 'unicode', u'#E9967A')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181529, unicode_181530))
# Adding element type (key, value) (line 999)
unicode_181531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1034, 4), 'unicode', u'darkseagreen')
unicode_181532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1034, 28), 'unicode', u'#8FBC8F')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181531, unicode_181532))
# Adding element type (key, value) (line 999)
unicode_181533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1035, 4), 'unicode', u'darkslateblue')
unicode_181534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1035, 28), 'unicode', u'#483D8B')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181533, unicode_181534))
# Adding element type (key, value) (line 999)
unicode_181535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1036, 4), 'unicode', u'darkslategray')
unicode_181536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1036, 28), 'unicode', u'#2F4F4F')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181535, unicode_181536))
# Adding element type (key, value) (line 999)
unicode_181537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1037, 4), 'unicode', u'darkslategrey')
unicode_181538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1037, 28), 'unicode', u'#2F4F4F')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181537, unicode_181538))
# Adding element type (key, value) (line 999)
unicode_181539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1038, 4), 'unicode', u'darkturquoise')
unicode_181540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1038, 28), 'unicode', u'#00CED1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181539, unicode_181540))
# Adding element type (key, value) (line 999)
unicode_181541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1039, 4), 'unicode', u'darkviolet')
unicode_181542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1039, 28), 'unicode', u'#9400D3')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181541, unicode_181542))
# Adding element type (key, value) (line 999)
unicode_181543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1040, 4), 'unicode', u'deeppink')
unicode_181544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1040, 28), 'unicode', u'#FF1493')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181543, unicode_181544))
# Adding element type (key, value) (line 999)
unicode_181545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1041, 4), 'unicode', u'deepskyblue')
unicode_181546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1041, 28), 'unicode', u'#00BFFF')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181545, unicode_181546))
# Adding element type (key, value) (line 999)
unicode_181547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1042, 4), 'unicode', u'dimgray')
unicode_181548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1042, 28), 'unicode', u'#696969')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181547, unicode_181548))
# Adding element type (key, value) (line 999)
unicode_181549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1043, 4), 'unicode', u'dimgrey')
unicode_181550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1043, 28), 'unicode', u'#696969')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181549, unicode_181550))
# Adding element type (key, value) (line 999)
unicode_181551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1044, 4), 'unicode', u'dodgerblue')
unicode_181552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1044, 28), 'unicode', u'#1E90FF')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181551, unicode_181552))
# Adding element type (key, value) (line 999)
unicode_181553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1045, 4), 'unicode', u'firebrick')
unicode_181554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1045, 28), 'unicode', u'#B22222')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181553, unicode_181554))
# Adding element type (key, value) (line 999)
unicode_181555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1046, 4), 'unicode', u'floralwhite')
unicode_181556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1046, 28), 'unicode', u'#FFFAF0')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181555, unicode_181556))
# Adding element type (key, value) (line 999)
unicode_181557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1047, 4), 'unicode', u'forestgreen')
unicode_181558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1047, 28), 'unicode', u'#228B22')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181557, unicode_181558))
# Adding element type (key, value) (line 999)
unicode_181559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1048, 4), 'unicode', u'fuchsia')
unicode_181560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1048, 28), 'unicode', u'#FF00FF')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181559, unicode_181560))
# Adding element type (key, value) (line 999)
unicode_181561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1049, 4), 'unicode', u'gainsboro')
unicode_181562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1049, 28), 'unicode', u'#DCDCDC')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181561, unicode_181562))
# Adding element type (key, value) (line 999)
unicode_181563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1050, 4), 'unicode', u'ghostwhite')
unicode_181564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1050, 28), 'unicode', u'#F8F8FF')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181563, unicode_181564))
# Adding element type (key, value) (line 999)
unicode_181565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1051, 4), 'unicode', u'gold')
unicode_181566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1051, 28), 'unicode', u'#FFD700')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181565, unicode_181566))
# Adding element type (key, value) (line 999)
unicode_181567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1052, 4), 'unicode', u'goldenrod')
unicode_181568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1052, 28), 'unicode', u'#DAA520')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181567, unicode_181568))
# Adding element type (key, value) (line 999)
unicode_181569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1053, 4), 'unicode', u'gray')
unicode_181570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1053, 28), 'unicode', u'#808080')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181569, unicode_181570))
# Adding element type (key, value) (line 999)
unicode_181571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1054, 4), 'unicode', u'green')
unicode_181572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1054, 28), 'unicode', u'#008000')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181571, unicode_181572))
# Adding element type (key, value) (line 999)
unicode_181573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1055, 4), 'unicode', u'greenyellow')
unicode_181574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1055, 28), 'unicode', u'#ADFF2F')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181573, unicode_181574))
# Adding element type (key, value) (line 999)
unicode_181575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1056, 4), 'unicode', u'grey')
unicode_181576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1056, 28), 'unicode', u'#808080')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181575, unicode_181576))
# Adding element type (key, value) (line 999)
unicode_181577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1057, 4), 'unicode', u'honeydew')
unicode_181578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1057, 28), 'unicode', u'#F0FFF0')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181577, unicode_181578))
# Adding element type (key, value) (line 999)
unicode_181579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1058, 4), 'unicode', u'hotpink')
unicode_181580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1058, 28), 'unicode', u'#FF69B4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181579, unicode_181580))
# Adding element type (key, value) (line 999)
unicode_181581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1059, 4), 'unicode', u'indianred')
unicode_181582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1059, 28), 'unicode', u'#CD5C5C')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181581, unicode_181582))
# Adding element type (key, value) (line 999)
unicode_181583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1060, 4), 'unicode', u'indigo')
unicode_181584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1060, 28), 'unicode', u'#4B0082')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181583, unicode_181584))
# Adding element type (key, value) (line 999)
unicode_181585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1061, 4), 'unicode', u'ivory')
unicode_181586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1061, 28), 'unicode', u'#FFFFF0')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181585, unicode_181586))
# Adding element type (key, value) (line 999)
unicode_181587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1062, 4), 'unicode', u'khaki')
unicode_181588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1062, 28), 'unicode', u'#F0E68C')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181587, unicode_181588))
# Adding element type (key, value) (line 999)
unicode_181589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1063, 4), 'unicode', u'lavender')
unicode_181590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1063, 28), 'unicode', u'#E6E6FA')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181589, unicode_181590))
# Adding element type (key, value) (line 999)
unicode_181591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1064, 4), 'unicode', u'lavenderblush')
unicode_181592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1064, 28), 'unicode', u'#FFF0F5')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181591, unicode_181592))
# Adding element type (key, value) (line 999)
unicode_181593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1065, 4), 'unicode', u'lawngreen')
unicode_181594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1065, 28), 'unicode', u'#7CFC00')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181593, unicode_181594))
# Adding element type (key, value) (line 999)
unicode_181595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1066, 4), 'unicode', u'lemonchiffon')
unicode_181596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1066, 28), 'unicode', u'#FFFACD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181595, unicode_181596))
# Adding element type (key, value) (line 999)
unicode_181597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1067, 4), 'unicode', u'lightblue')
unicode_181598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1067, 28), 'unicode', u'#ADD8E6')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181597, unicode_181598))
# Adding element type (key, value) (line 999)
unicode_181599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1068, 4), 'unicode', u'lightcoral')
unicode_181600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1068, 28), 'unicode', u'#F08080')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181599, unicode_181600))
# Adding element type (key, value) (line 999)
unicode_181601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1069, 4), 'unicode', u'lightcyan')
unicode_181602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1069, 28), 'unicode', u'#E0FFFF')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181601, unicode_181602))
# Adding element type (key, value) (line 999)
unicode_181603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1070, 4), 'unicode', u'lightgoldenrodyellow')
unicode_181604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1070, 28), 'unicode', u'#FAFAD2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181603, unicode_181604))
# Adding element type (key, value) (line 999)
unicode_181605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1071, 4), 'unicode', u'lightgray')
unicode_181606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1071, 28), 'unicode', u'#D3D3D3')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181605, unicode_181606))
# Adding element type (key, value) (line 999)
unicode_181607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1072, 4), 'unicode', u'lightgreen')
unicode_181608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1072, 28), 'unicode', u'#90EE90')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181607, unicode_181608))
# Adding element type (key, value) (line 999)
unicode_181609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1073, 4), 'unicode', u'lightgrey')
unicode_181610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1073, 28), 'unicode', u'#D3D3D3')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181609, unicode_181610))
# Adding element type (key, value) (line 999)
unicode_181611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1074, 4), 'unicode', u'lightpink')
unicode_181612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1074, 28), 'unicode', u'#FFB6C1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181611, unicode_181612))
# Adding element type (key, value) (line 999)
unicode_181613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1075, 4), 'unicode', u'lightsalmon')
unicode_181614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1075, 28), 'unicode', u'#FFA07A')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181613, unicode_181614))
# Adding element type (key, value) (line 999)
unicode_181615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1076, 4), 'unicode', u'lightseagreen')
unicode_181616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1076, 28), 'unicode', u'#20B2AA')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181615, unicode_181616))
# Adding element type (key, value) (line 999)
unicode_181617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1077, 4), 'unicode', u'lightskyblue')
unicode_181618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1077, 28), 'unicode', u'#87CEFA')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181617, unicode_181618))
# Adding element type (key, value) (line 999)
unicode_181619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1078, 4), 'unicode', u'lightslategray')
unicode_181620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1078, 28), 'unicode', u'#778899')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181619, unicode_181620))
# Adding element type (key, value) (line 999)
unicode_181621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1079, 4), 'unicode', u'lightslategrey')
unicode_181622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1079, 28), 'unicode', u'#778899')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181621, unicode_181622))
# Adding element type (key, value) (line 999)
unicode_181623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1080, 4), 'unicode', u'lightsteelblue')
unicode_181624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1080, 28), 'unicode', u'#B0C4DE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181623, unicode_181624))
# Adding element type (key, value) (line 999)
unicode_181625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1081, 4), 'unicode', u'lightyellow')
unicode_181626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1081, 28), 'unicode', u'#FFFFE0')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181625, unicode_181626))
# Adding element type (key, value) (line 999)
unicode_181627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1082, 4), 'unicode', u'lime')
unicode_181628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1082, 28), 'unicode', u'#00FF00')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181627, unicode_181628))
# Adding element type (key, value) (line 999)
unicode_181629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1083, 4), 'unicode', u'limegreen')
unicode_181630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1083, 28), 'unicode', u'#32CD32')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181629, unicode_181630))
# Adding element type (key, value) (line 999)
unicode_181631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1084, 4), 'unicode', u'linen')
unicode_181632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1084, 28), 'unicode', u'#FAF0E6')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181631, unicode_181632))
# Adding element type (key, value) (line 999)
unicode_181633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1085, 4), 'unicode', u'magenta')
unicode_181634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1085, 28), 'unicode', u'#FF00FF')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181633, unicode_181634))
# Adding element type (key, value) (line 999)
unicode_181635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1086, 4), 'unicode', u'maroon')
unicode_181636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1086, 28), 'unicode', u'#800000')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181635, unicode_181636))
# Adding element type (key, value) (line 999)
unicode_181637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1087, 4), 'unicode', u'mediumaquamarine')
unicode_181638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1087, 28), 'unicode', u'#66CDAA')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181637, unicode_181638))
# Adding element type (key, value) (line 999)
unicode_181639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1088, 4), 'unicode', u'mediumblue')
unicode_181640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1088, 28), 'unicode', u'#0000CD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181639, unicode_181640))
# Adding element type (key, value) (line 999)
unicode_181641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1089, 4), 'unicode', u'mediumorchid')
unicode_181642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1089, 28), 'unicode', u'#BA55D3')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181641, unicode_181642))
# Adding element type (key, value) (line 999)
unicode_181643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1090, 4), 'unicode', u'mediumpurple')
unicode_181644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1090, 28), 'unicode', u'#9370DB')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181643, unicode_181644))
# Adding element type (key, value) (line 999)
unicode_181645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1091, 4), 'unicode', u'mediumseagreen')
unicode_181646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1091, 28), 'unicode', u'#3CB371')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181645, unicode_181646))
# Adding element type (key, value) (line 999)
unicode_181647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1092, 4), 'unicode', u'mediumslateblue')
unicode_181648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1092, 28), 'unicode', u'#7B68EE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181647, unicode_181648))
# Adding element type (key, value) (line 999)
unicode_181649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1093, 4), 'unicode', u'mediumspringgreen')
unicode_181650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1093, 28), 'unicode', u'#00FA9A')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181649, unicode_181650))
# Adding element type (key, value) (line 999)
unicode_181651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1094, 4), 'unicode', u'mediumturquoise')
unicode_181652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1094, 28), 'unicode', u'#48D1CC')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181651, unicode_181652))
# Adding element type (key, value) (line 999)
unicode_181653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1095, 4), 'unicode', u'mediumvioletred')
unicode_181654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1095, 28), 'unicode', u'#C71585')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181653, unicode_181654))
# Adding element type (key, value) (line 999)
unicode_181655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1096, 4), 'unicode', u'midnightblue')
unicode_181656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1096, 28), 'unicode', u'#191970')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181655, unicode_181656))
# Adding element type (key, value) (line 999)
unicode_181657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1097, 4), 'unicode', u'mintcream')
unicode_181658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1097, 28), 'unicode', u'#F5FFFA')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181657, unicode_181658))
# Adding element type (key, value) (line 999)
unicode_181659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1098, 4), 'unicode', u'mistyrose')
unicode_181660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1098, 28), 'unicode', u'#FFE4E1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181659, unicode_181660))
# Adding element type (key, value) (line 999)
unicode_181661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1099, 4), 'unicode', u'moccasin')
unicode_181662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1099, 28), 'unicode', u'#FFE4B5')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181661, unicode_181662))
# Adding element type (key, value) (line 999)
unicode_181663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1100, 4), 'unicode', u'navajowhite')
unicode_181664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1100, 28), 'unicode', u'#FFDEAD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181663, unicode_181664))
# Adding element type (key, value) (line 999)
unicode_181665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1101, 4), 'unicode', u'navy')
unicode_181666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1101, 28), 'unicode', u'#000080')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181665, unicode_181666))
# Adding element type (key, value) (line 999)
unicode_181667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1102, 4), 'unicode', u'oldlace')
unicode_181668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1102, 28), 'unicode', u'#FDF5E6')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181667, unicode_181668))
# Adding element type (key, value) (line 999)
unicode_181669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1103, 4), 'unicode', u'olive')
unicode_181670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1103, 28), 'unicode', u'#808000')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181669, unicode_181670))
# Adding element type (key, value) (line 999)
unicode_181671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1104, 4), 'unicode', u'olivedrab')
unicode_181672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1104, 28), 'unicode', u'#6B8E23')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181671, unicode_181672))
# Adding element type (key, value) (line 999)
unicode_181673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1105, 4), 'unicode', u'orange')
unicode_181674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1105, 28), 'unicode', u'#FFA500')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181673, unicode_181674))
# Adding element type (key, value) (line 999)
unicode_181675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1106, 4), 'unicode', u'orangered')
unicode_181676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1106, 28), 'unicode', u'#FF4500')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181675, unicode_181676))
# Adding element type (key, value) (line 999)
unicode_181677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1107, 4), 'unicode', u'orchid')
unicode_181678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1107, 28), 'unicode', u'#DA70D6')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181677, unicode_181678))
# Adding element type (key, value) (line 999)
unicode_181679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1108, 4), 'unicode', u'palegoldenrod')
unicode_181680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1108, 28), 'unicode', u'#EEE8AA')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181679, unicode_181680))
# Adding element type (key, value) (line 999)
unicode_181681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1109, 4), 'unicode', u'palegreen')
unicode_181682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1109, 28), 'unicode', u'#98FB98')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181681, unicode_181682))
# Adding element type (key, value) (line 999)
unicode_181683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1110, 4), 'unicode', u'paleturquoise')
unicode_181684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1110, 28), 'unicode', u'#AFEEEE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181683, unicode_181684))
# Adding element type (key, value) (line 999)
unicode_181685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1111, 4), 'unicode', u'palevioletred')
unicode_181686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1111, 28), 'unicode', u'#DB7093')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181685, unicode_181686))
# Adding element type (key, value) (line 999)
unicode_181687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1112, 4), 'unicode', u'papayawhip')
unicode_181688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1112, 28), 'unicode', u'#FFEFD5')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181687, unicode_181688))
# Adding element type (key, value) (line 999)
unicode_181689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1113, 4), 'unicode', u'peachpuff')
unicode_181690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1113, 28), 'unicode', u'#FFDAB9')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181689, unicode_181690))
# Adding element type (key, value) (line 999)
unicode_181691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1114, 4), 'unicode', u'peru')
unicode_181692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1114, 28), 'unicode', u'#CD853F')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181691, unicode_181692))
# Adding element type (key, value) (line 999)
unicode_181693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1115, 4), 'unicode', u'pink')
unicode_181694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1115, 28), 'unicode', u'#FFC0CB')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181693, unicode_181694))
# Adding element type (key, value) (line 999)
unicode_181695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1116, 4), 'unicode', u'plum')
unicode_181696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1116, 28), 'unicode', u'#DDA0DD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181695, unicode_181696))
# Adding element type (key, value) (line 999)
unicode_181697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1117, 4), 'unicode', u'powderblue')
unicode_181698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1117, 28), 'unicode', u'#B0E0E6')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181697, unicode_181698))
# Adding element type (key, value) (line 999)
unicode_181699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1118, 4), 'unicode', u'purple')
unicode_181700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1118, 28), 'unicode', u'#800080')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181699, unicode_181700))
# Adding element type (key, value) (line 999)
unicode_181701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1119, 4), 'unicode', u'rebeccapurple')
unicode_181702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1119, 28), 'unicode', u'#663399')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181701, unicode_181702))
# Adding element type (key, value) (line 999)
unicode_181703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1120, 4), 'unicode', u'red')
unicode_181704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1120, 28), 'unicode', u'#FF0000')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181703, unicode_181704))
# Adding element type (key, value) (line 999)
unicode_181705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1121, 4), 'unicode', u'rosybrown')
unicode_181706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1121, 28), 'unicode', u'#BC8F8F')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181705, unicode_181706))
# Adding element type (key, value) (line 999)
unicode_181707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1122, 4), 'unicode', u'royalblue')
unicode_181708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1122, 28), 'unicode', u'#4169E1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181707, unicode_181708))
# Adding element type (key, value) (line 999)
unicode_181709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1123, 4), 'unicode', u'saddlebrown')
unicode_181710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1123, 28), 'unicode', u'#8B4513')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181709, unicode_181710))
# Adding element type (key, value) (line 999)
unicode_181711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1124, 4), 'unicode', u'salmon')
unicode_181712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1124, 28), 'unicode', u'#FA8072')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181711, unicode_181712))
# Adding element type (key, value) (line 999)
unicode_181713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1125, 4), 'unicode', u'sandybrown')
unicode_181714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1125, 28), 'unicode', u'#F4A460')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181713, unicode_181714))
# Adding element type (key, value) (line 999)
unicode_181715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1126, 4), 'unicode', u'seagreen')
unicode_181716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1126, 28), 'unicode', u'#2E8B57')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181715, unicode_181716))
# Adding element type (key, value) (line 999)
unicode_181717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1127, 4), 'unicode', u'seashell')
unicode_181718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1127, 28), 'unicode', u'#FFF5EE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181717, unicode_181718))
# Adding element type (key, value) (line 999)
unicode_181719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1128, 4), 'unicode', u'sienna')
unicode_181720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1128, 28), 'unicode', u'#A0522D')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181719, unicode_181720))
# Adding element type (key, value) (line 999)
unicode_181721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1129, 4), 'unicode', u'silver')
unicode_181722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1129, 28), 'unicode', u'#C0C0C0')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181721, unicode_181722))
# Adding element type (key, value) (line 999)
unicode_181723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1130, 4), 'unicode', u'skyblue')
unicode_181724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1130, 28), 'unicode', u'#87CEEB')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181723, unicode_181724))
# Adding element type (key, value) (line 999)
unicode_181725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1131, 4), 'unicode', u'slateblue')
unicode_181726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1131, 28), 'unicode', u'#6A5ACD')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181725, unicode_181726))
# Adding element type (key, value) (line 999)
unicode_181727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1132, 4), 'unicode', u'slategray')
unicode_181728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1132, 28), 'unicode', u'#708090')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181727, unicode_181728))
# Adding element type (key, value) (line 999)
unicode_181729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1133, 4), 'unicode', u'slategrey')
unicode_181730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1133, 28), 'unicode', u'#708090')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181729, unicode_181730))
# Adding element type (key, value) (line 999)
unicode_181731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1134, 4), 'unicode', u'snow')
unicode_181732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1134, 28), 'unicode', u'#FFFAFA')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181731, unicode_181732))
# Adding element type (key, value) (line 999)
unicode_181733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1135, 4), 'unicode', u'springgreen')
unicode_181734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1135, 28), 'unicode', u'#00FF7F')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181733, unicode_181734))
# Adding element type (key, value) (line 999)
unicode_181735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1136, 4), 'unicode', u'steelblue')
unicode_181736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1136, 28), 'unicode', u'#4682B4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181735, unicode_181736))
# Adding element type (key, value) (line 999)
unicode_181737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1137, 4), 'unicode', u'tan')
unicode_181738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1137, 28), 'unicode', u'#D2B48C')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181737, unicode_181738))
# Adding element type (key, value) (line 999)
unicode_181739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1138, 4), 'unicode', u'teal')
unicode_181740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1138, 28), 'unicode', u'#008080')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181739, unicode_181740))
# Adding element type (key, value) (line 999)
unicode_181741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1139, 4), 'unicode', u'thistle')
unicode_181742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1139, 28), 'unicode', u'#D8BFD8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181741, unicode_181742))
# Adding element type (key, value) (line 999)
unicode_181743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1140, 4), 'unicode', u'tomato')
unicode_181744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1140, 28), 'unicode', u'#FF6347')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181743, unicode_181744))
# Adding element type (key, value) (line 999)
unicode_181745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1141, 4), 'unicode', u'turquoise')
unicode_181746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1141, 28), 'unicode', u'#40E0D0')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181745, unicode_181746))
# Adding element type (key, value) (line 999)
unicode_181747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1142, 4), 'unicode', u'violet')
unicode_181748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1142, 28), 'unicode', u'#EE82EE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181747, unicode_181748))
# Adding element type (key, value) (line 999)
unicode_181749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1143, 4), 'unicode', u'wheat')
unicode_181750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1143, 28), 'unicode', u'#F5DEB3')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181749, unicode_181750))
# Adding element type (key, value) (line 999)
unicode_181751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1144, 4), 'unicode', u'white')
unicode_181752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1144, 28), 'unicode', u'#FFFFFF')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181751, unicode_181752))
# Adding element type (key, value) (line 999)
unicode_181753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1145, 4), 'unicode', u'whitesmoke')
unicode_181754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1145, 28), 'unicode', u'#F5F5F5')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181753, unicode_181754))
# Adding element type (key, value) (line 999)
unicode_181755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1146, 4), 'unicode', u'yellow')
unicode_181756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1146, 28), 'unicode', u'#FFFF00')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181755, unicode_181756))
# Adding element type (key, value) (line 999)
unicode_181757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1147, 4), 'unicode', u'yellowgreen')
unicode_181758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1147, 28), 'unicode', u'#9ACD32')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 14), dict_181462, (unicode_181757, unicode_181758))

# Assigning a type to the variable 'CSS4_COLORS' (line 999)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 999, 0), 'CSS4_COLORS', dict_181462)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
