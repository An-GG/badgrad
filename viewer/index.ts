type Neuron = {
    value: number
    value_before_activation: number
    bias: number
    input_layer_weights: number[] //first layer wont have obv
}

type Layer = {
    nodes: Neuron[]
}

type LayerValues = Neuron['value'][];

type Net = {
    layers: Layer[],
    nodes_per_layer: number[],
    activation_fn: (i:number) => number,
    derivative_activation_fn: (i:number) => number,
    training_metadata?: {
        avg_error: number
        rms_error: number
    }
}


type Netfile = {
    iterations: {[n:string]:{ layers: Net['layers'], training_metadata: Net['training_metadata'] } }
}

let net:Net;
let netfile:Netfile;

function calc_net_a0ada8b(net:Net, input:LayerValues, modifyOriginal?:boolean):Net {
    
    // NodeValue = avn_fn( NodeBias + (for each PNode in PrevLayer: PNode_k.value * Weight_k  /  #_of_Weights) )
    
    let isolated_net:Net = JSON.parse(JSON.stringify(net));

    // Calc First Layer Vals
    for (let nodeN = 0; nodeN < isolated_net.layers[0].nodes.length; nodeN++) {
        isolated_net.layers[0].nodes[nodeN].value_before_activation = isolated_net.layers[0].nodes[nodeN].bias + input[nodeN];
        let val = net.activation_fn(isolated_net.layers[0].nodes[nodeN].value_before_activation);
        isolated_net.layers[0].nodes[nodeN].value = val;
    } 

    // Calculate Everything Else
    let layerN = 0;
    for (let layer of isolated_net.layers) {
        
        // Skip first layer
        if (layerN == 0) { layerN++; continue; }
        let prevLayer = isolated_net.layers[layerN - 1];

        let nodeN = 0;
        for (let node of layer.nodes) {
            
            // Sum weight/node from prev layer 
            let weightedSum = 0;

            let wN = 0;
            for (let w of node.input_layer_weights) {
                weightedSum += (w * prevLayer.nodes[wN].value) / prevLayer.nodes.length;    
                wN++;
            }

            isolated_net.layers[layerN].nodes[nodeN].value_before_activation = node.bias + weightedSum; 
            isolated_net.layers[layerN].nodes[nodeN].value = net.activation_fn( node.bias + weightedSum ); 
            nodeN++;
        }
        layerN++;
    }

    if (modifyOriginal) { net.layers = isolated_net.layers; }    
    return isolated_net;
}


function sigmoid(x:number):number {
    return (1 / (1 + Math.pow(Math.E, -x)));
}

function maxIndex(arr:number[]):number {
    let i = 0;
    let max = -1;
    let ind = -1;
    for (let x of arr) {
        if (x > max) {
            max = x;
            ind = i;
        }
        i++
    }
    return ind;
}

function drawImage(ops: {
    image: number[],
    xPixels: number,
    pos: { x: number, y: number, width: number },
    ctx: CanvasRenderingContext2D
}) {

    let yPixels = ops.image.length / ops.xPixels;
    let pixleWidth = ops.pos.width / ops.xPixels;

    for (let y = 0; y < yPixels; y++) {
        for (let x = 0; x < ops.xPixels; x++) {
            let n = ( y * ops.xPixels ) + x;
            let pos = {x: ops.pos.x + (pixleWidth * x), y: ops.pos.y + (pixleWidth * y)};

            let w = ops.image[n];
            ops.ctx.fillStyle = "rgb("+w+","+w+","+w+")";
            ops.ctx.fillRect(pos.x, pos.y, pixleWidth, pixleWidth);
        }
    }

}

function rectContains(r:Rect, p:{x:number,y:number}): boolean {
    if (p.x >= r.pos.x && p.y >= r.pos.y) {
        if (p.x <= r.pos.x + r.size.w && p.y <= r.pos.y + r.size.h) {
            return true;
        }
    }
    return false;
}


let clickTargets : {box:Rect, func:()=>void}[] = [];

function addClickTarget(box: Rect, func:()=>void) {
    
    clickTargets.push({box:box, func:func});
}


function iterateClickTargets(clicked:{x:number, y:number}):boolean {
    let hit = false;
    for (let t of clickTargets) {
        if (rectContains(t.box, clicked)) {
            hit = true;
            t.func();
        }
    }
    return hit;
}

function scrunchPoint<P>(size: {w:number, h:number}, point:{x:number, y:number, [k:string]:any} & P, scrunch:{kW:number, kH:number}, anchor?:{x:number, y:number}): P {
    let out : typeof point = JSON.parse(JSON.stringify(point));
    anchor = anchor ? anchor : {x:size.w/2, y:size.h/2};
    out.x = anchor.x + ((point.x - anchor.x) * scrunch.kW)
    out.y = anchor.y + ((point.y - anchor.y) * scrunch.kH)
    return out;
}


function drawNet_cb(c:CanvasRenderingContext2D, mode: "LINE_ONLY" | "NODE_ONLY") {

    clickTargets = [];

    // netfile label
    c.fillStyle = 'white';
    c.font = "15px 'Roboto Mono'";
    c.fillText(net_name, 20, 50);

    let possiblePointSizes: number[] = [];
    let firstLayerVals = [];
    let lastLayerVals = [];

    let layerN = 0;
    for (let layer of net.layers) {
        let nodeN = 0;

        let maxbias = 1;
        let maxval = 1;
        for (let node of layer.nodes) {
            c.strokeStyle = "#ffffff";
            c.lineWidth = 3;
            
            let layerSpacing = (window.innerWidth) / (1 + net.layers.length);
            let nodeSpacing = (window.innerHeight) / (1 + layer.nodes.length);
            let rad = 25;
            let point = {
                x: (1+layerN) * layerSpacing,
                y: (1+nodeN) * nodeSpacing,
                size: (1/3) * (window.innerHeight / layer.nodes.length)
            }

            possiblePointSizes.push(point.size < 10 ? 10 : point.size);
            point.size = Math.min(...possiblePointSizes);
            point.size = point.size > 30 ? 30 : point.size;
            point = scrunchPoint(
                {w: window.innerWidth, h: window.innerHeight},
                point,
                {kW: 0.8, kH: 1});

            (node as any).point = point;
            if (mode == "NODE_ONLY") {


                c.lineWidth = point.size * 0.3;
                c.beginPath();
                let nodebias_rel = Math.round(Math.abs(node.bias*255)); //parseInt((sigmoid(Math.abs(node.bias) / maxbias)*255).toString());
                let nodeval_rel = Math.round(Math.abs(node.value*255));// parseInt((sigmoid(Math.abs(node.value) / maxval)*255).toString());
                
                if (nodebias_rel > 255) { nodebias_rel = 255; }
                if (nodeval_rel > 255) { nodeval_rel = 255; }

                if (node.bias > 0) {
                    c.strokeStyle = "#0022"+(nodebias_rel.toString(16).padStart(2,"0"))
                } else {
                    c.strokeStyle = "#"+(nodebias_rel.toString(16).padStart(2,"0")) + "2200";
                }

                if (node.value > 0) {
                    c.fillStyle = "#0000"+(nodeval_rel.toString(16).padStart(2,"0"));
                } else {
                    c.fillStyle = "#"+(nodeval_rel.toString(16).padStart(2,"0")) + "0000";
                }

                c.ellipse(point.x, point.y, point.size, point.size, 0, 0, 2*Math.PI);
                addClickTarget({pos:{x:point.x - (point.size / 2), y:point.y - (point.size / 2)}, size:{w:point.size,h:point.size}}, ()=>{
                    log(node, "node");
                });
                c.fill();
                c.stroke();

                // last layer vals
                if (layerN == net.layers.length - 1) {
                    c.fillStyle = 'white';
                    c.font = (point.size / 1.5) + "px 'Roboto Mono'";
                    c.fillText(node.value.toFixed(2), point.x + point.size * 1.5, point.y + (point.size/6));
                    lastLayerVals.push(node.value);
                }
                // first layer vals
                if (layerN == 0 && layer.nodes.length < 100) {
                    c.fillStyle = 'white';
                    c.font = (point.size / 1.5) + "px 'Roboto Mono'";
                    c.fillText(node.value.toFixed(2), point.x - point.size * 3, point.y + (point.size/6));
                    lastLayerVals.push(node.value);
                }
            }

            
            if (node.input_layer_weights && node.input_layer_weights.length > 0 && mode == "LINE_ONLY") {
                let maxweight = 0;
                for (let w of node.input_layer_weights) {
                    if (Math.abs(w) > maxweight) {
                        maxweight = Math.abs(w);
                    }
                }


                let wN = 0;
                let weightSpacing = (window.innerHeight) / (1 + node.input_layer_weights.length);
                for (let w of node.input_layer_weights) {
                    let prevpoint = (net.layers[layerN -1].nodes[wN] as any).point;
                    c.beginPath();

                    let weightval_rel = Math.round((w / maxweight) * 255);
                    if (w > 0) {
                        c.strokeStyle = "#0000ff";// "#0022"+weightval_rel.toString(16).padStart(2,"0");
                    } else {
                        c.strokeStyle = "#ff0000";//""#"+weightval_rel.toString(16).padStart(2,"0")+"2200";
                    }
                    c.lineWidth = Math.abs((sigmoid(w)) - 0.5) * point.size; 
                    c.moveTo(point.x, point.y);
                    c.lineTo(prevpoint.x, prevpoint.y);
                    c.stroke();
                    wN++;
                }
            }

            if (layerN == 0) {
                firstLayerVals.push(node.value - node.bias);
            }
            nodeN++;
        }
        layerN++;
    }

    drawImage({
        ctx: c,
        image: firstLayerVals,
        xPixels: 28,
        pos: { x: 20, y: 100, width: 100 },
    });

    (window as any).img_vals = firstLayerVals;

    // chosen answer label
    let chosen_ans = maxIndex(lastLayerVals);

    c.fillStyle = 'white';
    c.font = '15px Roboto Mono';
    if (chosen_ans != -1) {
        c.fillText("OUTPUT: " + chosen_ans.toString(), 20, window.innerHeight - 100);
        if (!net.training_metadata) { return; }
        c.fillText("RMSERR: " + net.training_metadata.rms_error.toString(), 20, window.innerHeight - 75);
        c.fillText("AVGERR: " + net.training_metadata.avg_error.toString(), 20, window.innerHeight - 50);
    }
}

type ListRange = {min:number, max:number};
function getRange(list:number[]): ListRange {
    return {
        max: Math.max(...list),
        min: Math.min(...list)
    }
}

function getNormalizedList(l:number[], customRange?:Partial<ListRange>):number[] {
    let out:number[] = [];
    let r = getRange(l);
    if (customRange) {
        if (typeof customRange.min == 'number') { r.min = customRange.min }
        if (typeof customRange.max == 'number') { r.max = customRange.max }
    }
    for (let v of l) {
        out.push((v-r.min) / (r.max - r.min));
    }
    return out;
}

type Plot = { list: number[], color: string, customRange?:Partial<ListRange> }
type Rect = {pos: {x:number, y:number}, size: {w:number, h:number}}
function drawGraph(c:CanvasRenderingContext2D, box:Rect, plots: Plot[], vline?:number) {
    
    c.fillStyle = 'white';
    c.fillRect(box.pos.x, box.pos.y, box.size.w, box.size.h);
    
    for (let p of plots) {
        let l = getNormalizedList(p.list, p.customRange);
        c.fillStyle = p.color;
        let vN = 0;
        for (let v of l) {
            let ptbox = [box.pos.x + ((vN/l.length) * box.size.w), box.pos.y + box.size.h - (v * box.size.h), 2, 2];
            c.fillRect(ptbox[0],ptbox[1],ptbox[2],ptbox[3]);
            (ptbox[0],ptbox[1],ptbox[2],ptbox[3]);
            vN++;
        }
    }
    if (typeof vline == 'number') {
        c.fillStyle = 'red';
        c.fillRect(box.pos.x + (vline * box.size.w), box.pos.y, 2, box.size.h);
    }

}




function drawNet(c:CanvasRenderingContext2D) {
    drawNet_cb(c, "LINE_ONLY");
    drawNet_cb(c, "NODE_ONLY");
}


let listenersAdded = false;
function setupCanvasContext(drawFunc:(ctx:CanvasRenderingContext2D)=>any) {
    let canvas = document.getElementById("canvas") as HTMLCanvasElement;
    let context = canvas.getContext('2d')!;

    // clear
    context.fillStyle = "black";
    context.fillRect(0,0,window.innerWidth,window.innerHeight);

    function onResize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        context.clearRect(0, 0, canvas.width, canvas.height);
        drawFunc(context); 
    }

    onResize();
    
    if (listenersAdded) { return; } else { listenersAdded = true; }
    window.addEventListener('resize', onResize);
    canvas.addEventListener('click', (e)=>{ event_click(e); });
}

function loadJSON(ref:string, cb:(response:string)=>any) {
    let xobj = new XMLHttpRequest();
    xobj.overrideMimeType("application/json");
    xobj.open('GET', ref, true);
    xobj.onreadystatechange = function() {
        if (xobj.readyState == 4 && xobj.status == 200) {
            cb(xobj.responseText);
        }
    }
    xobj.send(null);
}

function getNodesPerLayer(net:Netfile['iterations'][''] & {[k:string]:any}):number[] {
    let out = [];
    for (let l of net.layers) { out.push(l.nodes.length); }
    return out;
}



function reload_netfile() {
    if (urlparams.has('livegraph')) {
    
    } else {
     
        loadJSON(netfile_name, (r) => {
            netfile = JSON.parse(r);
            let isolated_net = netfile.iterations[net_name];
            (window as any).net_obj = net;
            (window as any).netfile_obj = netfile;
            
            // Gotta add activation
            net = {
                activation_fn: (i)=> { return (i > 0 ? i : 0); },
                derivative_activation_fn: (i) => { return (i > 0 ? 1 : 0); },
                layers: isolated_net.layers,
                nodes_per_layer: getNodesPerLayer(isolated_net),
                training_metadata: isolated_net.training_metadata
            }
            


            setupCanvasContext((c:CanvasRenderingContext2D) => {
                
                let graphsize = {w: c.canvas.width / 6, h: c.canvas.height / 8};
                let graphbox:Rect = { pos: {x: 0, y: c.canvas.height - (graphsize.h + (c.canvas.height * 0.2))}, size:graphsize };
                let plots:Plot[] = [];
                let plot_colors = ['red', 'green', 'blue'];
                let varnames = ['rms_error', 'avg_error'] as const;
                
                for (let net_nm in netfile.iterations) {
                    let md = netfile.iterations[net_nm].training_metadata;
                    if (!md) { continue; }
                    let vN = 0;
                    for (let v of varnames) {
                        if (plots.length == vN) { plots.push({ list:[], color:plot_colors[vN], customRange: { min:0 } }); }
                        plots[vN].list.push( md[v] );
                        vN++;
                    }
                }
                
                drawNet(c);
                drawGraph(c, graphbox, plots, current_net_index / net_names.length);

            });

            net_names = Object.keys(netfile.iterations);
            current_net_index = net_names.indexOf(net_name);

            slider.min = "0"; 
            slider.max = net_names.length.toString();
        });

    }
}

function drawAll(c:CanvasRenderingContext2D) {
    drawNet(c);
}


function next_netfile(fromSlider?:boolean) {
    current_net_index++;
    net_name = net_names[current_net_index];
    reload_netfile();
    var newurl = window.location.protocol + "//" + window.location.host + window.location.pathname + '?net=' + net_name;
    if (!fromSlider) { slider.value = current_net_index.toString(); }
    window.history.pushState({path:newurl},'',newurl);
    reload_netfile();
}

function sliderMove(to?:number) {
    if (typeof to == 'number') {
        current_net_index = to - 1;
        if (is_playing) {
            if (to >= net_names.length) {
                pause();
                return;
            }
        }
    } else {
        if (is_playing) { return; }
        pause();
        current_net_index = parseInt(slider.value) - 1;
    }
    next_netfile(typeof to != 'number');
}

let fps = 30;
let pb_rate = 30;
let time = 0;
let is_playing = false;
let pb_interval:NodeJS.Timeout;

function play() {
    pause();
    is_playing = true;
    time = Math.round(pb_rate / fps) * current_net_index;
    pb_interval = setInterval(()=>{
        time += (pb_rate / fps);
        sliderMove(Math.round(time));
    }, (1000) / fps); 
}

function pause() {
    clearInterval(pb_interval);
    is_playing = false;
}
function reset() {
    var newurl = window.location.protocol + "//" + window.location.host + window.location.pathname + '?net=0';
    window.location.href = newurl;
}

let netfile_name = 'netfile.json';
let net_name = "0";
let urlparams = new URLSearchParams(window.location.search);
if (urlparams.has('net')) {
    net_name = urlparams.get('net')!;
}

let net_names:string[] = [];
let current_net_index = 0;
let slider = window.document.getElementById('slider') as HTMLInputElement;

reload_netfile();







// Interaction

function event_click(e:MouseEvent) {
    let pos = {
        x: e.pageX,
        y: e.pageY
    }
    let hit = iterateClickTargets(pos);
    if (!hit) { 
        next_netfile();
    }
}

type objtype = "node"
function log(v:any, type?:objtype) {
    console.log(v);
    let out = "";
    switch(type) {
        case "node": 
            let val = v as Neuron;
            out+="===NODE===\n";
            out+="   VALUE: "+val.value.toPrecision(10)+"\n";
            out+="   BIAS:  "+val.bias.toPrecision(10)+"\n";
    }
    console.log(out);
}


(window as any).printImg = () => {
    let i = 0;
    let out = ""
    while (i < (window as any).img_vals.length) {
    out += (window as any).img_vals[i].toString().padStart(4, "_");
    if ((i + 1) % 28 == 0) { out += "\n"; }
    i++;
    }
    console.log(out);
}
