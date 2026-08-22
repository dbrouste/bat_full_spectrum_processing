(function () {
  const VERSION = "annotation-ui 1.3";
  const MODE_BUTTON_IDS = new Set(["new-chirp","add-point","move-point","delete-point","finish-chirp","delete-chirp"]);
  let savedViewport=null, interactionMode="navigation", lockUntil=0, restoring=false;

  function getPlot(){const r=document.getElementById("spectrogram");return r?r.querySelector(".js-plotly-plot"):null;}
  function viewportFromPlot(){const g=getPlot();if(!g||!g._fullLayout)return null;const x=g._fullLayout.xaxis,y=g._fullLayout.yaxis;if(!x||!y||!x.range||!y.range)return null;return{x:[+x.range[0],+x.range[1]],y:[+y.range[0],+y.range[1]]};}
  function snapshotViewport(){const v=viewportFromPlot();if(v)savedViewport=v;}
  function restoreViewportOnce(){if(!savedViewport||restoring)return;const g=getPlot();if(!g||!window.Plotly)return;restoring=true;Promise.resolve(Plotly.relayout(g,{"xaxis.range":savedViewport.x,"yaxis.range":savedViewport.y,"xaxis.autorange":false,"yaxis.autorange":false})).finally(()=>restoring=false);}
  function lockViewport(ms){snapshotViewport();if(!savedViewport)return;lockUntil=Math.max(lockUntil,Date.now()+ms);function loop(){if(Date.now()>lockUntil)return;restoreViewportOnce();requestAnimationFrame(loop);}requestAnimationFrame(loop);}
  function applyInteractionMode(){const g=getPlot();if(!g||!window.Plotly)return;snapshotViewport();const u=interactionMode==="navigation"?{clickmode:"none",dragmode:"zoom"}:{clickmode:"event+select",dragmode:false};Promise.resolve(Plotly.relayout(g,u)).then(restoreViewportOnce);const n=document.getElementById("interaction-navigation"),a=document.getElementById("interaction-annotation");if(n&&a){n.style.fontWeight=interactionMode==="navigation"?"700":"400";a.style.fontWeight=interactionMode==="annotation"?"700":"400";}}
  function setInteractionMode(m){if(m!=="navigation"&&m!=="annotation")return;lockViewport(1200);interactionMode=m;applyInteractionMode();}
  function addInteractionControls(){if(document.getElementById("interaction-mode-controls"))return;const g=document.getElementById("spectrogram");if(!g||!g.parentElement)return;const b=document.createElement("div");b.id="interaction-mode-controls";b.style.cssText="display:flex;gap:6px;align-items:center;margin:6px 0";const l=document.createElement("span");l.textContent="Graph mode:";l.style.fontWeight="600";const n=document.createElement("button");n.id="interaction-navigation";n.textContent="Navigation";n.onclick=()=>setInteractionMode("navigation");const a=document.createElement("button");a.id="interaction-annotation";a.textContent="Annotation";a.onclick=()=>setInteractionMode("annotation");b.append(l,n,a);g.parentElement.insertBefore(b,g);applyInteractionMode();}

  function styleNumericControls(){
    const floor=document.getElementById("db-floor");
    if(floor){floor.style.width="40px";floor.style.minWidth="40px";floor.style.maxWidth="40px";floor.style.boxSizing="border-box";}
    const max=document.getElementById("db-max");
    if(max){max.style.display="inline-block";max.style.visibility="visible";max.style.opacity="1";max.style.width="90px";max.style.minWidth="90px";max.style.maxWidth="90px";max.style.boxSizing="border-box";if(max.parentElement){max.parentElement.style.display="flex";max.parentElement.style.visibility="visible";max.parentElement.style.opacity="1";max.parentElement.style.flex="0 0 auto";}}
  }

  document.addEventListener("mousedown",e=>{const b=e.target.closest?e.target.closest("button"):null;if(b&&MODE_BUTTON_IDS.has(b.id))lockViewport(2500);},true);
  document.addEventListener("mousedown",e=>{if(interactionMode!=="annotation")return;const r=document.getElementById("spectrogram"),g=getPlot();if(r&&g&&r.contains(e.target)&&g.contains(e.target))lockViewport(3000);},true);
  document.addEventListener("click",e=>{const b=e.target.closest?e.target.closest("button"):null;if(b&&b.id==="new-chirp"){interactionMode="annotation";setTimeout(applyInteractionMode,0);}},true);

  function addVersionBadge(){let b=document.getElementById("viewport-guard-version");if(!b){b=document.createElement("div");b.id="viewport-guard-version";b.style.cssText="position:fixed;right:8px;bottom:6px;z-index:9999;font-size:10px;font-family:monospace;opacity:.45;pointer-events:none";document.body.appendChild(b);}b.textContent=VERSION;}
  function init(){addVersionBadge();addInteractionControls();styleNumericControls();if(!getPlot()){setTimeout(init,150);return;}applyInteractionMode();setInterval(styleNumericControls,500);}
  if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",init);else init();
})();
