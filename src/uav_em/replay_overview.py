"""Two-panel saved-flight display. No sensor, GP, optimizer, planner or LLM calls."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .equivalent_field import kernel, power_and_jacobian, normalized_expected_power


METHODS=('mixture_llm','mixture_numeric','gp_numeric')
LABELS={'gp_numeric':'GP估计＋数值规划','mixture_numeric':'等效场＋数值规划','mixture_llm':'等效场＋LLM辅助决策'}


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def saved_state(episode,frame):
    """Recover a cached before/after snapshot without recomputing evidence."""
    now=frame['time_seconds']
    completed=[d for d in episode['decisions'] if d['time_seconds']+.5<=now+1e-8]
    last=completed[-1] if completed else None
    pending=any(abs(d['time_seconds']-now)<1e-8 for d in episode['decisions'])
    observations=[p for p in episode['observations'] if p['time_seconds']<=now+1e-8]
    if 'hypotheses' in frame:
        models=deepcopy(frame['hypotheses'])
    elif len(observations)>1:
        models=deepcopy(episode['prediction_records'][len(observations)-2]['posterior'])
    else:
        models=[dict(id=n,family='persistence',std_db=sd,weight=.5,bases=[],scored_points=0)
                for n,sd in (('persistence',4.),('broad_residual',12.))]
    if 'hypotheses' not in frame:
        for event in episode['inferences']:
            if abs(event['time_seconds']-observations[-1]['time_seconds'])<1e-8 and event['time_seconds']+.5<=now+1e-8:
                models=deepcopy(event['bank_after_birth'])
    return models,observations,last,pending


def source_maps(models,xy,now,power):
    """Display-only evaluation of saved equivalent parameters, never physical truth."""
    means=[];terms=0
    for model in models:
        if model['family']=='persistence':mu=np.full(len(xy),power)
        else:
            mu=power_and_jacobian(kernel(model['bases'],xy,np.full(len(xy),now),model['reference_time_seconds']),model['peaks_dbm'])[0]
            terms+=len(xy)*len(model['bases'])
        means.append(mu)
    expected=sum(m['weight']*normalized_expected_power(mu,np.full(len(xy),m['std_db']),-80.) for m,mu in zip(models,means))
    actual=-80.+10.*np.log10(expected)
    indices=sorted([i for i,m in enumerate(models) if m['family']!='persistence'],key=lambda i:(-models[i]['weight'],models[i]['id']))[:2]
    return actual,[means[i] for i in indices],[models[i] for i in indices],terms


def numbers(array):
    return [float(x) if math.isfinite(float(x)) else None for x in np.asarray(array).ravel()]


def load_views(study,scene):
    newer=(study/'scenario.json').exists()
    cfg=read(study/'scenario.json' if newer else study/'scenarios'/f'{scene}.json')
    high=[(cfg['map']['columns']-1)*cfg['map']['resolution_m'],(cfg['map']['rows']-1)*cfg['map']['resolution_m']]
    views=[];inputs={};terms=0
    cfg_path=study/'scenario.json' if newer else study/'scenarios'/f'{scene}.json'
    inputs[str(cfg_path.relative_to(study))]=sha(cfg_path)
    for method in METHODS:
        episode_path=study/'runs'/(method if newer else f'{method}_{scene}')/'episode.json'
        if not episode_path.exists():continue
        ep=read(episode_path);inputs[str(episode_path.relative_to(study))]=sha(episode_path)
        if newer:
            grid_path=study/'evaluation'/f'{method}_grids.npz';pred_path=study/'predictions'/f'{method}.npz'
            pred=np.load(pred_path);truth=np.load(grid_path)['truth_dbm'];indices=pred['indices']
            inputs[str(pred_path.relative_to(study))]=sha(pred_path)
        else:
            grid_path=study/'evaluation'/f'{method}_{scene}_grids.npz'
            cached=np.load(grid_path);truth=cached['truth_dbm'];indices=cached['indices'];pred=None
        inputs[str(grid_path.relative_to(study))]=sha(grid_path)
        height,width=truth.shape[1:]
        xy=np.array([[x,y] for y in np.linspace(0,high[1],height) for x in np.linspace(0,high[0],width)])
        frames=[]
        for k,index in enumerate(indices):
            frame=ep['frames'][int(index)];now=frame['time_seconds']
            models,obs,last,pending=saved_state(ep,frame);power=obs[-1]['interference_dbm']
            actual,maps,top,work=source_maps(models,xy,now,power);terms+=work
            if newer:
                actual=np.asarray(pred['actual_dbm'][k]).ravel()
                # Existing frozen equivalent grids already use the same saved hypothesis order.
                maps=[np.asarray(pred['alternative_mean_dbm'][k,j]).ravel() for j in range(len(top))]
            if method=='gp_numeric':
                grid=np.asarray(frame['estimate_dbm']);support=np.asarray(frame['supported'],bool)
                ys=np.rint(np.linspace(0,grid.shape[0]-1,height)).astype(int)
                xs=np.rint(np.linspace(0,grid.shape[1]-1,width)).astype(int)
                gp=np.where(support,grid,np.nan)[np.ix_(ys,xs)].ravel()
                maps=[gp];actual=gp;top=[]
            basis=[]
            for model in top:
                basis.append([dict(position_m=(np.asarray(b['center_m'])+(now-model['reference_time_seconds'])*np.asarray(b['velocity_mps'])).tolist(),
                                   scale_m=b['scale_m'],velocity_mps=b['velocity_mps']) for b in model['bases']])
            sources=[dict(position_m=(np.asarray(s['center_m'])+(now-s['start_seconds'])*np.asarray(s['velocity_mps'])).tolist())
                for s in cfg['electromagnetic']['fields'] if s['start_seconds']<=now<s['end_seconds']]
            action='尚未发生重规划'
            proposal=None
            if last:
                plan=last['extra']['plan'];proposal=plan['waypoint_m'] if plan['action']=='via' else None
                action=('保留数值层选出的路线' if plan['action']=='keep' else
                        '新中间点已提交执行' if last['selected_id'].startswith('proposal') else '提议未被接受，执行数值路线')
            if pending:action+='；本帧为本次触发处理前'
            frames.append(dict(time_seconds=now,power_dbm=power,truth=numbers(truth[k]),actual=numbers(actual),
                hypotheses=[numbers(m) for m in maps],models=[dict(id=m['id'],family=m['family'],weight=m['weight'],scored_points=m['scored_points']) for m in top],
                bases=basis,true_sources=sources,constant_mass=sum(m['weight'] for m in models if m['family']=='persistence'),
                trail=[o['position_m'] for o in obs],position=frame['position_m'],route=frame['route_m'],
                triggers=[d['public']['request']['state']['position_m'] for d in ep['decisions'] if d['time_seconds']+.5<=now+1e-8],proposal=proposal,
                action=action,last_event_seconds=last['time_seconds'] if last else None))
        views.append(dict(method=method,label=LABELS[method],frames=frames,shape=[height,width],metrics=ep['metrics']))
    if not views:raise ValueError('no saved flights with cached truth grids')
    return dict(version='新事件闭环' if newer else '旧C1固定时刻对照：LLM仅辅助路径',
        bounds=high,goal=cfg['map']['goal_m'],buildings=cfg['map']['buildings'],
        altitude=cfg['aircraft']['altitude_m'],vertical_clearance=cfg['aircraft']['vertical_clearance_m'],views=views),inputs,terms


def render_gifs(destination,data,full_mixture=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from PIL import Image
    high=data['bounds']
    for view in data['views']:
        images=[];height,width=view['shape']
        for frame in view['frames']:
            fig,axes=plt.subplots(1,2,figsize=(11,5),sharex=True,sharey=True)
            fig.subplots_adjust(top=.77,bottom=.19,left=.055,right=.86,wspace=.12)
            predicted=frame['actual'] if full_mixture else frame['hypotheses'][0] if frame['hypotheses'] else [None]*(height*width)
            right_title='Full mixture: current-time expected power' if full_mixture else 'Strongest source hypothesis (one possible world)'
            titles=['True interference (evaluation only)', 'Local GP estimate' if view['method']=='gp_numeric' else right_title]
            for side,(ax,field,title) in enumerate(zip(axes,(frame['truth'],predicted),titles)):
                z=np.asarray([math.nan if v is None else v for v in field]).reshape(height,width)
                cmap=plt.get_cmap('YlOrRd').copy();cmap.set_bad('#c7ccd2')
                im=ax.imshow(np.ma.masked_invalid(z),origin='lower',extent=(0,high[0],0,high[1]),vmin=-100,vmax=-55,cmap=cmap)
                for b in data['buildings']:
                    color='#303030' if b['height_m']+data['vertical_clearance']>=data['altitude'] else '#b9cddd'
                    ax.add_patch(Polygon(b['footprint_m'],color=color,zorder=3))
                ax.plot(*np.asarray(frame['trail']).T,color='#185adb',lw=2.2,zorder=4)
                if frame['route']:ax.plot(*np.asarray(frame['route']).T,'--',color='#007e91',lw=1.8,zorder=4)
                ax.plot(*frame['position'],'ko',ms=5,zorder=6);ax.plot(*data['goal'],'b*',ms=11,zorder=6)
                for p in frame['triggers']:ax.plot(*p,'o',mfc='none',mec='#b100a5',ms=8,zorder=5)
                if frame['proposal']:ax.plot(*frame['proposal'],'D',mfc='none',mec='#b100a5',ms=7,zorder=5)
                if side==0:
                    for source in frame['true_sources']:ax.plot(*source['position_m'],'*',color='black',ms=11,zorder=5)
                elif frame['bases'] and not full_mixture:
                    for basis in frame['bases'][0]:
                        ax.plot(*basis['position_m'],'x',color='#8a2387',ms=9,mew=2,zorder=5)
                ax.set_title(title,fontsize=10);ax.set_xlim(0,high[0]);ax.set_ylim(0,high[1]);ax.set_xlabel('x (m)')
            weight=frame['models'][0]['weight'] if frame['models'] else None
            label='all retained hypotheses; original evidence weights' if full_mixture else f"source-hypothesis weight={weight:.2g}" if weight is not None else 'no equivalent-source hypothesis yet' if view['method']!='gp_numeric' else 'gray = unsupported'
            fig.suptitle(f"{view['method']} | t={frame['time_seconds']:.1f}s | point measurement={frame['power_dbm']:.1f} dBm\n"
                f"{label}; constant-power mass={frame['constant_mass']:.1%}",fontsize=11)
            legend='Blue: flown   Dashed teal: actual remaining route   Black star: true source'
            explanation='Right: weighted expectation of alternative worlds, not all sources added together. Planner evaluates future arrival times.'
            if not full_mixture:
                legend+='   Purple x: equivalent basis'
                explanation='Right: one source explanation for location inspection. Actual planner uses the full mixture, including constant power.'
            fig.text(.06,.06,legend+'\n'+explanation,fontsize=9)
            cb=fig.add_axes([.89,.22,.015,.5])
            bar=fig.colorbar(im,cax=cb)
            bar.set_label('dBm: higher = stronger',fontsize=8,labelpad=3)
            cb.tick_params(labelsize=8)
            fig.canvas.draw();images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()));plt.close(fig)
        suffix='full_mixture' if full_mixture else 'overview'
        path=destination/f"{view['method']}_{suffix}.gif"
        images[0].save(path,save_all=True,append_images=images[1:],duration=500,loop=0)


def render_overview(study,scene='dynamic',view='source'):
    """New display artifacts alongside unchanged original results and GIFs."""
    if not study.is_dir():raise FileNotFoundError(study)
    if view not in ('source','full-mixture'):raise ValueError(view)
    data,inputs,terms=load_views(study,scene)
    full_mixture=view=='full-mixture'
    suffix='full_mixture' if full_mixture else 'overview'
    if full_mixture:data['views']=[v for v in data['views'] if v['method']!='gp_numeric']
    destination=study/'replay'/suffix
    destination.mkdir(exist_ok=False)
    payload=json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    page=PAGE.replace('__DATA__',payload).replace('__DEFAULT_MODE__','actual' if full_mixture else 'source').replace('__GIF_SUFFIX__',suffix)
    if full_mixture:
        page=page.replace('右边默认显示一张有源的等效解释，方便看清预测位置。','右边默认显示完整假设池在当前时刻的功率期望，包含功率延续及预测不确定性。不同假设是替代环境，不是所有源同时存在。路径评分另按沿途未来到达时刻计算。')
        page=page.replace('<p id="downloads"></p>','<p><a href="../overview/index.html">另看单张等效源解释与GP参考</a></p><p id="downloads"></p>')
    (destination/'index.html').write_text(page)
    render_gifs(destination,data,full_mixture)
    for rel,expected in inputs.items():
        if sha(study/rel)!=expected:raise ValueError('display changed an original input')
    generated={p.name:sha(p) for p in destination.iterdir() if p.is_file()}
    with (destination/'display_provenance.json').open('x') as stream:
        json.dump(dict(kind='display_only',source_version=data['version'],input_sha256=inputs,
            display_basis_receiver_terms=terms,new_truth_queries=0,new_gp_queries=0,new_fits=0,new_flights=0,new_llm_calls=0,
            source_sha256=sha(Path(__file__)),original_inputs_unchanged=True,
            right_default='current-time full-pool expected linear power including uncertainty, displayed in dBm; not simultaneous sources' if full_mixture else 'highest-weight source-containing alternative, conditionally displayed; not the full planner mixture',
            generated_sha256=generated),stream,ensure_ascii=False,indent=2)
    print('DISPLAY',destination/'index.html',flush=True)


PAGE='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>真实环境与等效预测：飞行对照</title>
<style>body{font:16px/1.6 system-ui;margin:24px auto;max-width:1250px;padding:0 16px;color:#223}h1{font-size:25px}select,button,input{font:inherit;padding:5px}button{cursor:pointer}canvas{width:100%;background:#f6f7f9;border:1px solid #d5dce5}#status,#meaning{padding:9px 13px;background:#f1f5fb;border-radius:6px;margin:10px 0}.muted{color:#586578}a{color:#165c9a}details{margin:15px 0}table{border-collapse:collapse}td,th{border:1px solid #cdd;padding:7px}#seek{width:65%;vertical-align:middle}</style>
<h1>真实环境、等效预测与UAV行动</h1><p id="version"></p>
<p>左边是真实干扰，仅供事后对照；右边默认显示一张有源的等效解释，方便看清预测位置。两边的蓝线都是实际已飞航迹，青色虚线都是已提交的剩余路线。</p>
<label>方法：<select id="method"></select></label>　<label>右侧显示：<select id="mode"><option value="source">主要等效源解释</option><option value="other">另一张等效解释</option><option value="actual">实际规划使用的混合预测</option></select></label>
<div id="meaning"></div><canvas id="canvas" width="1200" height="560"></canvas>
<p><button id="play">暂停</button> <input id="seek" type="range" min="0" value="0"> <span id="time"></span></p><div id="status"></div>
<p class="muted">黑星＝真实源（仅左图）；紫叉＝当前这张解释的等效基元；黑圆＝UAV；蓝星＝goal；紫圈＝已处理的触发点；紫菱形＝最近中间点提议，可能未被执行。灰色＝未知或尚无源解释。颜色统一：越红功率越强。</p>
<details><summary>三个名称是什么意思？为什么实际混合图可能是同一颜色？</summary><table><tr><th>内部名称</th><th>含义</th></tr><tr><td>gp_numeric</td><td>用局部GP预测，数值方法选路径，不调用LLM。</td></tr><tr><td>mixture_numeric</td><td>保留多个等效场解释，数值规则组织拟合，数值方法选路径。</td></tr><tr><td>mixture_llm</td><td>等效场＋LLM辅助路径；旧C1推断仍由数值组织，新事件版才由LLM组织推断与路径。</td></tr></table><p>mixture表示多个替代环境解释；numeric表示数值规则/优化器。某个“功率延续”解释认为各处等于最近一次读数，当它占大部分权重时，混合图就接近均匀。这不是全图实测。默认右图单独显示一张有空间结构的源解释，其全池权重可能很低，不能误认为系统确定相信它。可切换到“实际规划使用的混合预测”核对。</p></details>
<p id="downloads"></p><p class="muted">仅重画已有航迹和缓存真值，没有重飞、重拟合或新增LLM调用。动画为压缩播放，不代表实机响应速度。</p>
<script>const D=__DATA__;
const M=document.querySelector('#method'),S=document.querySelector('#mode'),R=document.querySelector('#seek'),C=document.querySelector('#canvas'),G=C.getContext('2d');let playing=true;
S.value='__DEFAULT_MODE__';
D.views.forEach((v,i)=>{let o=document.createElement('option');o.value=i;o.textContent=v.label+'（'+v.method+'）';M.append(o)});
document.querySelector('#version').textContent='数据版本：'+D.version;
document.querySelector('#downloads').innerHTML=D.views.map(v=>'<a href="'+v.method+'___GIF_SUFFIX__.gif">'+v.label+'：双图GIF</a>').join(' · ');
const pct=x=>x>0&&x<.0001?(x*100).toExponential(1)+'%':(100*x).toFixed(2)+'%';
function color(v){if(v===null||!Number.isFinite(v))return '#c7ccd2';const a=[[255,255,204],[255,237,160],[254,217,118],[254,178,76],[253,141,60],[252,78,42],[227,26,28],[189,0,38],[128,0,38]];let t=Math.max(0,Math.min(1,(v+100)/45))*(a.length-1),i=Math.min(a.length-2,Math.floor(t)),f=t-i;return 'rgb('+a[i].map((x,j)=>Math.round(x+(a[i+1][j]-x)*f)).join(',')+')'}
function draw(){const v=D.views[+M.value];R.max=v.frames.length-1;let f=v.frames[+R.value],gp=v.method==='gp_numeric',idx=S.value==='other'?1:0;let field=S.value==='actual'?f.actual:f.hypotheses[idx];let model=f.models[idx];let actual=S.value==='actual';if(gp){field=f.hypotheses[0];model=null}G.clearRect(0,0,C.width,C.height);const w=480,h=360,y=62,xs=[65,655];
function pos(p,x){return [x+p[0]/D.bounds[0]*w,y+h-p[1]/D.bounds[1]*h]}
function line(points,x,col,dash){if(!points||!points.length)return;G.beginPath();points.forEach((p,i)=>{let q=pos(p,x);i?G.lineTo(...q):G.moveTo(...q)});G.strokeStyle=col;G.lineWidth=3;G.setLineDash(dash||[]);G.stroke();G.setLineDash([])}
function mark(p,x,kind,col){let [a,b]=pos(p,x);G.strokeStyle=col;G.fillStyle=col;G.lineWidth=2;G.beginPath();if(kind==='x'){G.moveTo(a-6,b-6);G.lineTo(a+6,b+6);G.moveTo(a-6,b+6);G.lineTo(a+6,b-6);G.stroke()}else if(kind==='star'){for(let i=0;i<10;i++){let angle=-Math.PI/2+i*Math.PI/5,r=i%2?4:9;let q=[a+Math.cos(angle)*r,b+Math.sin(angle)*r];i?G.lineTo(...q):G.moveTo(...q)}G.closePath();G.fill()}else{G.arc(a,b,kind==='ring'?8:5,0,Math.PI*2);kind==='ring'?G.stroke():G.fill()}}
[f.truth,field||Array(v.shape[0]*v.shape[1]).fill(null)].forEach((z,n)=>{let x=xs[n],ny=v.shape[0],nx=v.shape[1];for(let j=0;j<ny;j++)for(let i=0;i<nx;i++){G.fillStyle=color(z[j*nx+i]);G.fillRect(x+i*w/nx,y+h-(j+1)*h/ny,w/nx+.5,h/ny+.5)}D.buildings.forEach(b=>{G.fillStyle=b.height_m+D.vertical_clearance>=D.altitude?'#303030':'#b9cddd';G.beginPath();b.footprint_m.forEach((p,i)=>{let q=pos(p,x);i?G.lineTo(...q):G.moveTo(...q)});G.closePath();G.fill()});line(f.trail,x,'#185adb');line(f.route,x,'#007e91',[9,6]);mark(D.goal,x,'star','#185adb');f.triggers.forEach(p=>mark(p,x,'ring','#b100a5'));mark(f.position,x,'dot','#111');if(f.proposal){let [a,b]=pos(f.proposal,x);G.strokeStyle='#b100a5';G.strokeRect(a-5,b-5,10,10)}if(n===0)f.true_sources.forEach(s=>mark(s.position_m,x,'star','#111'));else if(!actual&&!gp&&f.bases[idx])f.bases[idx].forEach(b=>mark(b.position_m,x,'x','#8a2387'));G.strokeStyle='#607080';G.lineWidth=1;G.strokeRect(x,y,w,h);G.fillStyle='#223';G.font='18px system-ui';G.textAlign='center';G.fillText(n===0?'真实干扰场（仅评价）':gp?'GP局部估计':actual?'实际规划混合预测':'等效源解释（'+(idx+1)+'）',x+w/2,30);G.font='13px system-ui';for(let i=0;i<=4;i++){G.fillText(Math.round(D.bounds[0]*i/4),x+w*i/4,y+h+22);G.fillText(Math.round(D.bounds[1]*i/4),x-25,y+h-h*i/4)}G.fillText('x / m',x+w/2,y+h+44)});
for(let j=0;j<180;j++){G.fillStyle=color(-55-j/179*45);G.fillRect(1149,90+j*1.5,16,2)}G.fillStyle='#223';G.font='13px system-ui';G.textAlign='left';G.fillText('-55',1170,94);G.fillText('-100',1170,365);G.fillText('dBm',1145,390);
document.querySelector('#time').textContent='t = '+f.time_seconds.toFixed(1)+' 秒';document.querySelector('#status').textContent='当前位置功率：'+f.power_dbm.toFixed(1)+' dBm；'+f.action+'。'+(f.last_event_seconds===null?'':'最近已处理事件：'+f.last_event_seconds+'秒。');
document.querySelector('#meaning').textContent=gp?'右图灰色为GP无支持区域，不是安全区。':actual?'右图是所有解释按原权重混合的期望功率。功率延续解释占 '+pct(f.constant_mass)+'；均匀颜色不等于周围均已测量。':model?'右图只显示这一张源解释，便于观察预测位置。它占全池 '+pct(model.weight)+'，获得 '+model.scored_points+' 条出生后点测；实际规划仍使用全池，功率延续占 '+pct(f.constant_mass)+'。':'尚未形成这张等效源解释，灰色表示没有可显示的源地图；系统仍有功率延续等保底解释。';}
M.onchange=()=>{R.value=0;S.value='__DEFAULT_MODE__';draw()};S.onchange=draw;R.oninput=()=>{playing=false;document.querySelector('#play').textContent='播放';draw()};document.querySelector('#play').onclick=()=>{playing=!playing;document.querySelector('#play').textContent=playing?'暂停':'播放'};setInterval(()=>{if(playing){let v=D.views[+M.value];R.value=(+R.value+1)%v.frames.length;draw()}},650);draw();</script></html>'''
