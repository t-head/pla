#!/usr/bin/env python3
# Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Add (or refresh) the intracard-CP panel in flashkda-e2e-vs-fla.html.

Data injection lives in merge_kimi_cp_into_report.py; this script only owns the
markup + render code, so the two can be re-run independently.  Both the section
and its JS are wrapped in sentinel comments, which makes re-running idempotent:
an existing panel is replaced rather than duplicated.

The panel is deliberately built to read as an experiment, not a scoreboard:
  * CP gain is shown as a slow|fast decay pair, never pooled -- pre_scan's
    zero-state early exit makes the cost data-dependent.
  * the control is the same binary on the same card with PLA_FLASHKDA_CP=0.
  * the planner census is shown so "CP declines for H>12" is visibly measured.
  * the production-mix rows are presented as a null control, not as a win.

Usage:
  python3 add_cp_panel_to_report.py --html .../flashkda-e2e-vs-fla.html
"""
import argparse
import re
import sys

SEC_BEGIN = "<!-- CP-PANEL:BEGIN -->"
SEC_END = "<!-- CP-PANEL:END -->"
JS_BEGIN = "/* CP-PANEL:JS:BEGIN */"
JS_END = "/* CP-PANEL:JS:END */"
FOOT_BEGIN = "<!-- CP-FOOT:BEGIN -->"
FOOT_END = "<!-- CP-FOOT:END -->"

SECTION = SEC_BEGIN + """
  <section class="panel" id="cp-panel" style="border-top:3px solid #a58bff">
    <div class="panel-head">
      <div>
        <h2>Intracard CP (in-card sequence parallelism)<span style="font-size:12px;color:#a58bff;font-weight:600;letter-spacing:.4px">added 2026-08-19</span></h2>
        <p class="sub" id="cp-meta">&mdash;</p>
      </div>
      <div class="controls">
        <label>Decay<select id="cp-decay"></select></label>
        <label>State<select id="cp-mode"><option value="all">All</option><option value="no_state">No state</option><option value="with_state">With state</option></select></label>
      </div>
    </div>

    <p class="sub" style="margin:0 0 12px">
      The first two panels pushed <code>flashkdapro</code> to the point where "every chain is fast enough"; the remaining lagging points are all <strong>too few chains, each too long</strong>:
      the grid has only <code>N&times;H</code> CTAs, CUs sit idle, and the recurrence of a 30k-token chain must run serially to the end.
      <strong>CP splits one chain into several segments and advances them in parallel</strong>, at the cost of a three-pass pipeline:
      <code>pre_scan</code> (B chain with zero init + M chain with identity init and <code>v:=0</code>, the M/B duality of cuLA) &rarr;
      <code>kernel3_cp</code> serial merge (<code>C<sub>new</sub><sup>T</sup> = C<sub>old</sub><sup>T</sup>M<sub>st</sub> + B<sup>T</sup></code>, zero transposes throughout) &rarr;
      segmented parallel K2. The total work is about <strong>2.3 passes</strong> of recurrence, so it can only win when "the chain count is far below the CU count" &mdash;&mdash; hence the strict gating.
    </p>
    <aside class="warning" style="margin:0 0 14px"><div>&#9888;</div><div><strong>CP cost is data dependent and must be read as a slow|fast pair of decay bounds</strong><span id="cp-bound-note" style="display:block;margin-top:6px;font-weight:400"></span></div></aside>

    <section class="cards" style="margin-bottom:14px">
      <div class="card"><div class="k">shapes accepted by the planner</div><div class="v" id="cp-cov">&mdash;</div><div class="s" id="cp-cov-detail">&mdash;</div></div>
      <div class="card"><div class="k">CP gain (slow &vert; fast)</div><div class="v accent" id="cp-gain">&mdash;</div><div class="s">CP off &divide; CP on, same binary, same card</div></div>
      <div class="card"><div class="k">vs FLA (current decay)</div><div class="v" id="cp-spshift">&mdash;</div><div class="s" id="cp-loss-shift">&mdash;</div></div>
      <div class="card"><div class="k">production-mix null-control</div><div class="v" id="cp-null">&mdash;</div><div class="s" id="cp-null-detail">&mdash;</div></div>
    </section>

    <h3 style="margin:18px 0 8px;font-size:15px">Planner census: on which shapes does CP actually engage</h3>
    <p class="sub" style="margin:0 0 8px">Whether CP engages is <strong>answered by the kernel's own <code>flashkda_cp_plan</code></strong>, not by replicating the gate in Python, so this table cannot drift from the dispatch under test.</p>
    <div class="table-wrap" style="max-height:none"><table class="summary-table">
      <thead><tr><th>Group</th><th>H</th><th>Unique shapes</th><th>CP engage</th><th>Share</th><th>T range engaged</th><th>Verdict</th></tr></thead>
      <tbody id="cp-census-body"></tbody>
    </table></div>

    <h3 style="margin:22px 0 8px;font-size:15px">Shapes where CP engages: decay bounds &times; state</h3>
    <div class="table-wrap" style="max-height:none"><table class="summary-table">
      <thead><tr><th>Decay</th><th>State</th><th>N</th><th>gain geo</th><th>gain median</th><th>gain range</th><th>vs FLA (off&rarr;on)</th><th>Lagging points</th><th>CP off median</th><th>CP on median</th><th>FLA median</th></tr></thead>
      <tbody id="cp-decay-body"></tbody>
    </table></div>

    <h3 style="margin:22px 0 8px;font-size:15px">Per-shape gain (x axis T, y axis CP off &divide; CP on)</h3>
    <div id="cp-scatter" style="margin-bottom:8px"></div>
    <div class="legend" id="cp-legend"></div>

    <h3 style="margin:22px 0 8px;font-size:15px">Details of shapes where CP engages</h3>
    <p class="sub" id="cp-visible">&mdash;</p>
    <div class="table-wrap"><table>
      <thead><tr><th data-ck="decay">Decay</th><th data-ck="T_total">T</th><th data-ck="n_seg">n_seg</th><th data-ck="max_seg_tiles">max seg tiles</th><th data-ck="mode">State</th><th data-ck="ppu_cpoff_median_ms">CP off ms</th><th data-ck="ppu_median_ms">CP on ms</th><th data-ck="fla_median_ms">FLA ms</th><th data-ck="cp_gain">gain</th><th data-ck="speedup_cpoff">sp off</th><th data-ck="speedup">sp on</th><th data-ck="ppu_cv">on CV</th></tr></thead>
      <tbody id="cp-detail-body"></tbody>
    </table></div>

    <h3 style="margin:22px 0 8px;font-size:15px">Production-mix null-control (same sampling as the 1200 points above)</h3>
    <p class="sub" style="margin:0 0 8px">These points use <code>torch.empty</code> inputs and the same 150-shape sampling as the dataset above. The goal is <strong>not</strong> to show CP gains (the gate rejects the vast majority of shapes outright), but to prove that <strong>wiring CP into the default dispatch slowed down nothing that should not change</strong>.</p>
    <div class="table-wrap" style="max-height:none"><table class="summary-table">
      <thead><tr><th>Group</th><th>N</th><th>engaged</th><th>gain geo</th><th>gain range</th><th>vs FLA (off&rarr;on)</th><th>Lagging points off&rarr;on</th></tr></thead>
      <tbody id="cp-mix-body"></tbody>
    </table></div>

    <aside class="warning" style="margin-top:18px"><div>&#9888;</div><div><strong>CP cannot fix the lagging points left over from the previous panel</strong><span id="cp-limits" style="display:block;margin-top:6px;font-weight:400"></span></div></aside>
  </section>
""" + SEC_END

RENDER = JS_BEGIN + """
/* ===== intracard CP panel ===== */
(function(){
  const blob = JSON.parse(document.getElementById('report-data').textContent);
  const S = blob.cp_summary, FR = blob.cp_focus_rows || [], MR = blob.cp_rows || [];
  const panel = document.getElementById('cp-panel');
  if(!S || !panel){ if(panel) panel.style.display='none'; return; }
  const $ = id => document.getElementById(id);
  const fx = (v,n)=> (v==null||isNaN(v))?'—':Number(v).toFixed(n==null?3:n);
  const DC_COLOR = {slow:'#f2b23c', mid:'#68a9ff', fast:'#4de3c1', empty:'#8b93a7'};
  const DC_LABEL = {slow:'slow (pre_scan runs in full)', mid:'mid', fast:'fast (early exit)', empty:'empty'};
  const HOF = {1:96,2:48,4:24,8:12};
  const okFR = FR.filter(r=>r.status==='ok');

  /* --- meta --- */
  const m = S.meta||{};
  const scopeBits = [];
  if (Array.isArray(m.allow_tp) && m.allow_tp.length) scopeBits.push('tp∈{' + m.allow_tp.join(',') + '}');
  if (m.max_batch != null) scopeBits.push('B≤' + m.max_batch);
  const scope = scopeBits.length ? ' · <strong>scope</strong> ' + scopeBits.join(' · ') : '';
  $('cp-meta').innerHTML = 'pla <code>'+(m.pla_branch||'?')+'</code> @ <code>'+(m.pla_commit||'?')+'</code>'
    + ' · ' + (m.host||'') + ' · collected ' + (m.created_at||'')
    + ' · control: <strong>same binary / same card / same process</strong> <code>PLA_FLASHKDA_CP=0</code> (getenv is read on every launch)'
    + scope;

  /* --- decay selector --- */
  const regimes = (S.focus&&S.focus.regimes)||[];
  $('cp-decay').innerHTML = regimes.map(d=>'<option value="'+d+'">'+(DC_LABEL[d]||d)+'</option>').join('')
    || '<option value="all">All</option>';

  /* --- bound note --- */
  const gb = (S.focus&&S.focus.gain_bounds)||{};
  const pair = regimes.map(d=>'<strong>'+fx(gb[d],3)+'×</strong> ('+d+')').join(' … ');
  $('cp-bound-note').innerHTML =
    'The transfer chain of pre_scan has an <strong>exact</strong> early exit on zero state (<code>ZCHK=8</code>, sign-masked OR-reduce + atomicOr + dual-CTA barrier): '
    + 'with fast decay M<sub>i</sub> underflows within a few chunks and pre_scan pays only a small part; with slow decay it never underflows and pre_scan pays a full pass. '
    + 'Real traffic falls between the two, so this panel <strong>always gives a pair of bounds</strong> rather than a single number: CP gain = ' + (pair||'—') + '. '
    + 'Pooling slow / fast into one geometric mean forms no bound at all, so the table is split by decay and not merged.';

  /* --- census --- */
  (function(){
    const cs = S.census||[];
    $('cp-census-body').innerHTML = cs.map(c=>{
      const eng = c.engaged_shapes||0, tot = c.total_shapes||0;
      const verdict = eng>0
        ? '<span class="good">engage</span> (N=1 and T long enough)'
        : '<span style="color:#8b93a7">planner rejects all</span> (H&gt;<code>CP_MAX_H</code>=12)';
      return '<tr'+(eng>0?' style="background:rgba(165,139,255,.10)"':'')+'>'
        +'<td><strong>tp'+c.tp+'</strong></td><td>'+(c.H||HOF[c.tp]||'—')+'</td>'
        +'<td>'+tot+'</td><td>'+eng+'</td>'
        +'<td>'+(tot?(100*eng/tot).toFixed(1)+'%':'—')+'</td>'
        +'<td>'+(eng?c.engaged_T_min+' … '+c.engaged_T_max:'—')+'</td>'
        +'<td>'+verdict+'</td></tr>';
    }).join('') || '<tr><td colspan="7">no census data</td></tr>';
    const hit = cs.filter(c=>(c.engaged_shapes||0)>0);
    const tot = cs.reduce((a,c)=>a+(c.total_shapes||0),0);
    const eng = cs.reduce((a,c)=>a+(c.engaged_shapes||0),0);
    $('cp-cov').textContent = eng + ' / ' + tot;
    $('cp-cov-detail').innerHTML = hit.length
      ? 'only '+hit.map(c=>'tp'+c.tp+'(H='+(c.H||HOF[c.tp])+')').join(', ')+' hit'
      : 'no hits';
  })();

  /* --- decay × mode summary --- */
  (function(){
    const rows=[];
    (S.focus&&S.focus.by_decay||[]).forEach(d=>{
      const pack=(label,a)=> '<tr'+(d.decay==='slow'?' style="background:rgba(242,178,60,.08)"':'')+'>'
        +'<td><span class="dot" style="display:inline-block;width:8px;height:8px;border-radius:50%;background:'+(DC_COLOR[d.decay]||'#68a9ff')+';margin-right:6px"></span>'+d.decay+'</td>'
        +'<td>'+label+'</td><td>'+a.count+'</td>'
        +'<td class="'+(a.gain_geo>=1.02?'good':(a.gain_geo<=0.98?'bad':''))+'"><strong>'+fx(a.gain_geo,4)+'×</strong></td>'
        +'<td>'+fx(a.gain_med,4)+'×</td>'
        +'<td>'+fx(a.gain_min,3)+'× .. '+fx(a.gain_max,3)+'×</td>'
        +'<td>'+fx(a.sp_off_geo,3)+'× → '+fx(a.sp_on_geo,3)+'×</td>'
        +'<td>'+a.losses_off+' → '+a.losses_on+'</td>'
        +'<td>'+fx(a.ppu_off_med_ms,3)+' ms</td><td>'+fx(a.ppu_on_med_ms,3)+' ms</td>'
        +'<td>'+fx(a.fla_med_ms,3)+' ms</td></tr>';
      if(d.overall && d.overall.count) rows.push(pack('<strong>All</strong>', d.overall));
      (d.by_mode||[]).forEach(a=>{ if(a.count) rows.push(pack(a.mode, a)); });
    });
    $('cp-decay-body').innerHTML = rows.join('') || '<tr><td colspan="11">no data</td></tr>';
  })();

  /* --- mix null control --- */
  (function(){
    const mix = S.mix||{};
    const pack=(label,a,extra)=> '<tr><td>'+label+'</td><td>'+a.count+'</td><td>'+(extra==null?a.engaged_count:extra)+'</td>'
      +'<td class="'+(a.gain_geo>=0.98?'good':'bad')+'">'+fx(a.gain_geo,4)+'×</td>'
      +'<td>'+fx(a.gain_min,3)+'× .. '+fx(a.gain_max,3)+'×</td>'
      +'<td>'+fx(a.sp_off_geo,3)+'× → '+fx(a.sp_on_geo,3)+'×</td>'
      +'<td>'+a.losses_off+' → '+a.losses_on+'</td></tr>';
    const out=[];
    if(mix.overall&&mix.overall.count) out.push(pack('<strong>All</strong>', mix.overall));
    (mix.by_tp||[]).forEach(a=>out.push(pack('tp'+a.tp+' (H='+(a.H||HOF[a.tp])+')', a)));
    (mix.buckets||[]).forEach(a=>out.push(pack(a.label==='CP engaged'?'subset accepted by the gate':'subset rejected by the gate', a)));
    $('cp-mix-body').innerHTML = out.join('') || '<tr><td colspan="7">no data</td></tr>';
    const o = mix.overall||{};
    $('cp-null').textContent = fx(o.gain_geo,4)+'×';
    $('cp-null-detail').innerHTML = (o.count||0)+' points, worst '+fx(o.gain_min,3)+'×; '
      + 'of which the gate accepted '+(mix.engaged_count||0)+' and rejected '+(mix.plain_count||0)+' (rejected means bit-identical, the same kernel path)';
  })();

  /* --- cards that depend on the selected decay --- */
  function selDecay(){ return $('cp-decay').value; }
  function decayAgg(){
    return ((S.focus&&S.focus.by_decay)||[]).find(d=>d.decay===selDecay());
  }
  function updateCards(){
    $('cp-gain').innerHTML = regimes.map(d=>fx(gb[d],3)+'×').join(' <span style="color:#8b93a7">|</span> ')||'—';
    const d = decayAgg(); const a = (d&&d.overall)||{};
    $('cp-spshift').textContent = fx(a.sp_off_geo,3)+'× → '+fx(a.sp_on_geo,3)+'×';
    $('cp-loss-shift').textContent = 'decay='+selDecay()+', <1× points '+(a.losses_off==null?'—':a.losses_off)+' → '+(a.losses_on==null?'—':a.losses_on);
  }

  /* --- scatter --- */
  function cfilt(){
    const dc=selDecay(), md=$('cp-mode').value;
    return okFR.filter(r=>(dc==='all'||r.decay===dc)&&(md==='all'||r.mode===md));
  }
  function drawScatter(rs){
    const W=980,H=280,PL=54,PR=14,PT=14,PB=34;
    if(!rs.length){ $('cp-scatter').innerHTML='<p class="sub">no data</p>'; return; }
    const xs=rs.map(r=>r.T_total), ys=rs.map(r=>r.cp_gain);
    const lx=v=>Math.log10(Math.max(v,1)), ly=v=>Math.log2(Math.max(v,0.05));
    const x0=Math.min.apply(null,xs.map(lx)), x1=Math.max.apply(null,xs.map(lx));
    const y0=Math.min.apply(null,ys.map(ly).concat([-0.6])), y1=Math.max.apply(null,ys.map(ly).concat([0.6]));
    const sx=v=>PL+(lx(v)-x0)/((x1-x0)||1)*(W-PL-PR);
    const sy=v=>PT+(y1-ly(v))/((y1-y0)||1)*(H-PT-PB);
    let g='';
    g+='<line x1="'+PL+'" y1="'+sy(1)+'" x2="'+(W-PR)+'" y2="'+sy(1)+'" stroke="#f2b23c" stroke-dasharray="5 4" stroke-width="1.2"/>';
    g+='<text x="'+(W-PR-4)+'" y="'+(sy(1)-6)+'" fill="#f2b23c" font-size="11" text-anchor="end">1× = CP breaks even</text>';
    [0.7,1,1.5,2,3].forEach(t=>{ if(t>=Math.pow(2,y0)&&t<=Math.pow(2,y1)){
      g+='<line x1="'+PL+'" y1="'+sy(t)+'" x2="'+(W-PR)+'" y2="'+sy(t)+'" stroke="#232a3a" stroke-width="1"/>';
      g+='<text x="'+(PL-8)+'" y="'+(sy(t)+4)+'" fill="#8b93a7" font-size="10" text-anchor="end">'+t+'×</text>';}});
    rs.forEach(r=>{
      const c=DC_COLOR[r.decay]||'#68a9ff';
      g+='<circle cx="'+sx(r.T_total).toFixed(1)+'" cy="'+sy(r.cp_gain).toFixed(1)+'" r="3.2" fill="'+c+'"'
       + ' opacity="'+(r.mode==='with_state'?0.55:0.9)+'"'
       + (r.cp_gain<1?' stroke="#ff6b6b" stroke-width="1"':'')
       + '><title>T='+r.T_total+' n_seg='+r.n_seg+' ['+r.mode+' / '+r.decay+']\\ngain '+fx(r.cp_gain,3)+'×\\nCP off '+fx(r.ppu_cpoff_median_ms,3)+' ms → CP on '+fx(r.ppu_median_ms,3)+' ms\\nFLA '+fx(r.fla_median_ms,3)+' ms</title></circle>';
    });
    g+='<text x="'+((PL+W-PR)/2)+'" y="'+(H-6)+'" fill="#8b93a7" font-size="11" text-anchor="middle">T_total (tokens per sequence, log₁₀)</text>';
    $('cp-scatter').innerHTML='<svg viewBox="0 0 '+W+' '+H+'" style="width:100%;height:auto;background:#0f1420;border:1px solid #1e2637;border-radius:10px">'+g+'</svg>';
    $('cp-legend').innerHTML = regimes.map(d=>'<span><i class="dot" style="background:'+(DC_COLOR[d]||'#68a9ff')+'"></i>decay '+d+'</span>').join('')
      + '<span style="color:#8b93a7">solid = no state, translucent = with state · red ring = CP is actually slower</span>';
  }

  /* --- detail table --- */
  let ck='cp_gain', casc=false;
  function drawTable(rs){
    const arr=rs.slice().sort((a,b)=>{const va=a[ck],vb=b[ck];
      const c=(typeof va==='number'&&typeof vb==='number')?va-vb:String(va).localeCompare(String(vb));
      return casc?c:-c;});
    $('cp-visible').textContent = arr.length+' / '+okFR.length+' points (click a header to sort)';
    $('cp-detail-body').innerHTML = arr.slice(0,400).map(r=>
      '<tr><td>'+r.decay+'</td><td>'+r.T_total+'</td><td>'+r.n_seg+'</td><td>'+r.max_seg_tiles+'</td><td>'+r.mode+'</td>'
      +'<td>'+fx(r.ppu_cpoff_median_ms,3)+'</td><td>'+fx(r.ppu_median_ms,3)+'</td><td>'+fx(r.fla_median_ms,3)+'</td>'
      +'<td class="'+(r.cp_gain>=1.02?'good':(r.cp_gain<=0.98?'bad':''))+'"><strong>'+fx(r.cp_gain,3)+'×</strong></td>'
      +'<td>'+fx(r.speedup_cpoff,3)+'×</td>'
      +'<td class="'+(r.speedup>=1?'good':'bad')+'">'+fx(r.speedup,3)+'×</td>'
      +'<td>'+fx(r.ppu_cv,3)+'</td></tr>').join('')
      + (arr.length>400?'<tr><td colspan="12" style="color:#8b93a7">… showing only the first 400 rows</td></tr>':'');
  }

  /* --- limits / honesty box --- */
  (function(){
    const x = S.xcheck||{};
    const cs = (S.census||[]).filter(c=>(c.engaged_shapes||0)>0).map(c=>'tp'+c.tp);
    $('cp-limits').innerHTML =
      'The points still &lt;1× after relaxing the dvsplit gate in the previous panel are all <strong>H=12 ragged varlen with one segment holding about 92%</strong> (e.g. <code>[30209, 2559]</code>). '
      + 'The gate landed here explicitly does <strong>not engage</strong> for <code>varlen &amp;&amp; H==12 &amp;&amp; N&ge;2</code> (<code>cp_min_seq_len_for_shape</code> returns <code>INT32_MAX</code>), '
      + 'so <strong>these points are not fixed by CP</strong>, and this panel does not claim to fix them. The reason is not laziness: under varlen K1/K2 already parallelize per segment, and the chain count <code>N&times;H&times;4</code> usually fills 39 CUs, '
      + 'so stacking 2.3 passes of CP only makes it slower; absorbing this kind of skew needs splitting <strong>by token rather than by sequence</strong> + associative merging of state across segments, which is engineering of another order of magnitude. '
      + '<br>The range where CP actually takes effect is that one row of the census table: <strong>single-sequence long prefill</strong> of ' + (cs.join(', ')||'—') + ' (fixed N=1, T &ge; <code>CP_FIXED_H12_MIN_SEQ_LEN</code>=16384).'
      + '<br><span style="color:#8b93a7">Cross-machine consistency check: CP-off on this machine paired with the dataset at the top of the report (another machine) over '
      + (x.paired||0) + ' points, ratio geomean ' + fx(x.geo_ratio,3) + '× (range ' + fx(x.min_ratio,3) + '..' + fx(x.max_ratio,3) + '×). '
      + 'This only shows the two machines are of the same magnitude and is <strong>not a CP conclusion</strong> -- every CP before/after comes from the same card and the same process.</span>';
  })();

  function refresh(){ const rs=cfilt(); updateCards(); drawScatter(rs); drawTable(rs); }
  ['cp-decay','cp-mode'].forEach(id=>$(id).addEventListener('change',refresh));
  document.querySelectorAll('th[data-ck]').forEach(th=>th.addEventListener('click',()=>{
    const k=th.dataset.ck; if(ck===k) casc=!casc; else {ck=k; casc=true;} drawTable(cfilt());}));
  refresh();
})();
""" + JS_END

FOOTNOTE = (FOOT_BEGIN
            + "<br>Intracard CP A/B data (2026-08-19) is embedded in <code>report-data.cp_focus_rows</code>"
              " / <code>cp_rows</code> / <code>cp_summary</code>; collection script "
              "<code>kimi_k3_cp_rows.py</code>, injection scripts "
              "<code>merge_kimi_cp_into_report.py</code> + <code>add_cp_panel_to_report.py</code>."
            + FOOT_END)


def splice(text, begin, end, payload, anchor):
    """Idempotent insert: replace an existing sentinel block, else insert at anchor."""
    if begin in text and end in text:
        i, j = text.index(begin), text.index(end) + len(end)
        return text[:i] + payload + text[j:], "replaced"
    if anchor not in text:
        raise SystemExit(f"anchor not found: {anchor!r}")
    i = text.index(anchor)
    return text[:i] + payload + text[i:], "inserted"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", required=True)
    args = ap.parse_args()

    with open(args.html, encoding="utf-8") as fh:
        html = fh.read()
    n0 = len(html)

    html, a1 = splice(html, SEC_BEGIN, SEC_END, SECTION, "\n  <footer>")
    html, a2 = splice(html, JS_BEGIN, JS_END, RENDER, "\n</script>\n</body>")
    html, a3 = splice(html, FOOT_BEGIN, FOOT_END, FOOTNOTE, "</footer>")

    # The panel must not disturb the data blob.
    m = re.search(r'<script id="report-data"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        raise SystemExit("report-data blob vanished")
    import json
    json.loads(m.group(1))

    with open(args.html, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"section {a1}, render {a2}, footnote {a3}; "
          f"HTML {n0} -> {len(html)} bytes, data blob still parses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
