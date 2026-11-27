"""The page shell: CSS, script and header, in the Language Exposition's look (dark by default, big text, one font)."""

CSS = r"""
@font-face{font-family:'Recursive Variable';font-style:normal;font-display:swap;font-weight:300 1000;src:url(assets/recursive-latin-crsv-normal.woff2) format('woff2-variations')}
:root{--bg:#f7f9fb;--ink:#17202a;--ink-soft:#2b3b48;--muted:#4b5a68;--card:#fff;--line:#d5dfe5;--accent:#0f766e;--accent-ink:#fff;--chip-bg:#eef3f5;
 --header-bg:#ffffffee;--th-bg:#eef3f5;--row-alt:#f6f9fb;--shadow:0 1px 2px #0001,0 6px 24px #0000000f;--maxw:1648px;--prose:74ch}
:root[data-theme="dark"]{--bg:#0e141a;--ink:#e6edf3;--ink-soft:#c9d5de;--muted:#9fb1bf;--card:#151e26;--line:#2a3946;--accent:#2dd4bf;--accent-ink:#06201c;
 --header-bg:#0e141aee;--th-bg:#1b2731;--chip-bg:#1b2731;--row-alt:#121b22;--shadow:0 1px 2px #0006,0 8px 30px #0000004d}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:76px;-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:18px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
header.top{position:sticky;top:0;z-index:50;background:var(--header-bg);backdrop-filter:blur(8px);border-bottom:1px solid var(--line);display:flex;align-items:center;gap:6px 18px;padding:10px 20px;flex-wrap:wrap}
header.top .brand{color:var(--ink);text-decoration:none;font:700 21px Georgia,"Times New Roman",serif}
header.top nav{display:flex;gap:2px 4px;flex-wrap:wrap;flex:1;justify-content:flex-end}
header.top nav a{color:var(--ink-soft);text-decoration:none;font-size:16px;padding:6px 10px;border-radius:8px}
header.top nav a:hover{background:var(--chip-bg);color:var(--accent)}
.themebtn{border:1px solid var(--line);background:var(--chip-bg);color:var(--ink);border-radius:999px;padding:6px 14px;font:inherit;font-size:15px;cursor:pointer}
.themebtn:hover{border-color:var(--accent);color:var(--accent)}
:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
.wrap{max-width:var(--maxw);margin:0 auto;padding:0 24px}
h1{font-family:Georgia,"Times New Roman",serif;font-size:48px;line-height:1.1;margin:48px 0 14px;letter-spacing:-.02em}
h2{font-family:Georgia,"Times New Roman",serif;font-size:36px;line-height:1.2;margin:72px 0 8px;letter-spacing:-.01em;border-top:1px solid var(--line);padding-top:28px}
h3{font-family:Georgia,"Times New Roman",serif;font-size:29px;margin:56px 0 6px;color:var(--ink)}
h4{font-size:21px;margin:30px 0 6px}
p,li{max-width:var(--prose)}
p{margin:10px 0}
.lede{font-size:21px;color:var(--ink-soft)}
a{color:var(--accent)}
code{font:14.5px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;background:var(--chip-bg);border-radius:5px;padding:1px 6px;overflow-wrap:anywhere}
a.ref{text-decoration:none}
a.ref code{color:var(--accent);white-space:normal;word-break:break-all}
a.ref:hover code{background:var(--accent);color:var(--accent-ink)}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0}
.chips a{border:1px solid var(--line);border-radius:999px;padding:4px 13px;font-size:15px;text-decoration:none;background:var(--chip-bg);color:var(--ink-soft)}
.chips a:hover{border-color:var(--accent);color:var(--accent)}
.tablewrap{overflow-x:auto;max-width:100%}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:15.5px;line-height:1.5}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left;vertical-align:top}
th{background:var(--th-bg);position:sticky;top:0}
tr:nth-child(even) td{background:var(--row-alt)}
tr:target td{background:var(--accent);color:var(--accent-ink)}
tr:target td a,tr:target td code{color:var(--accent-ink)}
table.objects td:nth-child(1),table.trans td:nth-child(1),table.pats td:nth-child(1),table.sortable td:nth-child(1){min-width:150px}
table.sortable th{cursor:pointer}
table.sortable th:hover{color:var(--accent)}
input.filter{font:inherit;font-size:16px;padding:7px 12px;border:1px solid var(--line);border-radius:10px;background:var(--card);color:var(--ink);width:min(420px,100%)}
details{margin:12px 0;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:8px 14px}
details>summary{cursor:pointer;font-weight:600;color:var(--accent);font-size:17px}
.mdiag{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin:14px 0;overflow:auto;box-shadow:var(--shadow)}
.mdiag svg{display:block;max-width:100%;height:auto;margin:0 auto}
.mdiag .mlt{display:none}
:root[data-theme="light"] .mdiag .mlt{display:block}
:root[data-theme="light"] .mdiag .mdk{display:none}
.mdiag foreignObject div,.mdiag foreignObject span,.mdiag foreignObject p{font:inherit;line-height:1.35;margin:0;padding:0;max-width:none;letter-spacing:normal}
.mdiag .node.linked{cursor:pointer}
.mdiag .node.linked:hover rect,.mdiag .node.linked:hover path,.mdiag .node.linked:hover polygon{stroke-width:4px}
.guarantees-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:18px;margin-top:14px}
.gcard{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:6px 18px 14px;box-shadow:var(--shadow);min-width:0}
.gcard h4{margin:14px 0 4px}
.gcard .mdiag{box-shadow:none;margin:8px 0}
footer{border-top:1px solid var(--line);margin-top:80px;padding:24px 0 60px;color:var(--muted);font-size:15px}
@media(max-width:700px){h1{font-size:36px}h2{font-size:29px}body{font-size:17px}.guarantees-grid{grid-template-columns:1fr}}
"""

JS = r"""
(function(){
  var root=document.documentElement;
  function store(k,v){try{if(v===undefined)return localStorage.getItem(k);localStorage.setItem(k,v);}catch(e){return null;}}
  var s=store("mapal-theme"); if(s==="light"||s==="dark")root.setAttribute("data-theme",s);
  var tb=document.getElementById("themebtn");
  function paint(){tb.textContent=root.getAttribute("data-theme")==="light"?"Dark theme":"Light theme";}
  paint();
  tb.addEventListener("click",function(){var n=root.getAttribute("data-theme")==="light"?"dark":"light";root.setAttribute("data-theme",n);store("mapal-theme",n);paint();});
  // diagram nodes jump to their row or section
  document.querySelectorAll(".mdiag[data-links]").forEach(function(box){
    var links=JSON.parse(box.getAttribute("data-links"));
    box.querySelectorAll("g.node").forEach(function(g){
      var m=/^flowchart-(.+)-\d+$/.exec(g.id||""); if(!m||!links[m[1]])return;
      g.classList.add("linked"); g.setAttribute("tabindex","0"); g.setAttribute("role","link");
      var go=function(){location.hash=links[m[1]];};
      g.addEventListener("click",go); g.addEventListener("keydown",function(e){if(e.key==="Enter")go();});
    });
  });
  // a link into a closed fold opens it
  function openTarget(){
    var id=decodeURIComponent(location.hash.slice(1)); if(!id)return;
    var el=document.getElementById(id); if(!el)return;
    for(var p=el.parentElement;p;p=p.parentElement){if(p.tagName==="DETAILS")p.open=true;}
    el.scrollIntoView({block:"center"});
  }
  addEventListener("hashchange",openTarget); openTarget();
  // sortable, filterable index
  document.querySelectorAll("table.sortable").forEach(function(t){
    var body=t.tBodies[0], rows=[].slice.call(body.rows);
    var f=document.createElement("input"); f.className="filter"; f.placeholder="Filter the index…"; f.setAttribute("aria-label","Filter the index");
    t.parentElement.parentElement.insertBefore(f,t.parentElement);
    f.addEventListener("input",function(){var q=f.value.toLowerCase();rows.forEach(function(r){r.style.display=r.textContent.toLowerCase().indexOf(q)<0?"none":"";});});
    [].slice.call(t.tHead.rows[0].cells).forEach(function(th,i){
      var dir=1; th.addEventListener("click",function(){
        rows.sort(function(a,b){return dir*a.cells[i].textContent.localeCompare(b.cells[i].textContent);}); dir=-dir;
        rows.forEach(function(r){body.appendChild(r);});
      });
    });
  });
})();
"""

NAV = [("data", "Data"), ("memory", "Memory"), ("components", "Components"), ("guarantees", "Guarantees"), ("index", "Index")]


def shell(body: str, digest: str, stats: str, diagram_css: str = "") -> str:
    nav = "".join(f'<a href="#{a}">{t}</a>' for a, t in NAV)
    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="color-scheme" content="dark light">
<title>Mapal — Architecture Map</title>
<meta name="description" content="The Mapal compiler as data: categories, transformations and guarantees, every item linked to the code that defines it.">
<link rel="preload" as="font" type="font/woff2" crossorigin href="assets/recursive-latin-crsv-normal.woff2">
<style>{CSS}</style>
<style>{diagram_css}</style>
</head>
<body>
<header class="top">
  <a class="brand" href="#top">Mapal · Architecture Map</a>
  <nav aria-label="Sections">{nav}</nav>
  <button type="button" class="themebtn" id="themebtn" aria-label="Switch between dark and light theme">Light theme</button>
</header>
<main id="top" class="wrap">
<h1>The compiler as data</h1>
<p class="lede">Mapal's compiler seen as the data that defines it: kinds of data (categories), the things and arrows inside them, and the transformations between them. {stats} Every item links to the line of code that defines it.</p>
{body}
</main>
<footer><div class="wrap">Generated from <code>docs/architecture/map.json</code> by <code>docs/tools/archmap.py</code>; <code>--check</code> keeps this page true. Older, more detailed notes: <a href="architecture/categorical-model.md">categorical-model.md</a>, <a href="architecture/deduced-queries.md">deduced-queries.md</a>.<br><!-- archmap:hash={digest} --><small>content hash {digest}</small></div></footer>
<script>{JS}</script>
</body>
</html>
"""
