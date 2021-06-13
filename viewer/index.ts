
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
    training_metadata: {
        error: number
    }
}


let net:Net;


function calc_net_a0ada8b(net:Net, input:LayerValues, modifyOriginal?:boolean):Net {
    
    // NodeValue = avn_fn( NodeBias + (for each PNode in PrevLayer: PNode_k.value * Weight_k  /  #_of_Weights) )
    
    let isolated_net:Net = JSON.parse(JSON.stringify(net));

    // Gotta add activation
    isolated_net = {
        activation_fn: (i)=> { return (i > 0 ? i : 0); },
        derivative_activation_fn: (i) => { return (i > 0 ? 1 : 0); },
        layers: isolated_net.layers,
        nodes_per_layer: isolated_net.nodes_per_layer,
        training_metadata: isolated_net.training_metadata
    }


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

function drawNet(c:CanvasRenderingContext2D, mode: "LINE_ONLY" | "NODE_ONLY") {


    // netfile label
    c.fillStyle = 'white';
    c.font = "30px Arial";
    c.fillText(netfile_name, 20, 50);

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
                c.fill();
                c.stroke();

                // last layer vals
                if (layerN == net.layers.length - 1) {
                    c.fillStyle = 'white';
                    c.font = (point.size*2) + "px Arial";
                    c.fillText(node.value.toFixed(2), point.x + point.size * 3, point.y + (point.size/2));
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
                    let prevpoint = {
                        x: layerN * layerSpacing,
                        y: (1+wN) * weightSpacing
                    }
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
    console.log(chosen_ans);

    c.fillStyle = 'white';
    c.font = '30px Arial';
    if (chosen_ans != -1) {
        c.fillText(chosen_ans.toString(), 20, 250);
        c.fillText(net.training_metadata.error.toString(), 20, 300);
    }
}



function draw(c:CanvasRenderingContext2D) {
    drawNet(c, "LINE_ONLY");
    drawNet(c, "NODE_ONLY");
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

function reload_netfile() {
    loadJSON(netfile_name, (r) => {
        net = JSON.parse(r);
        console.log(net.layers[0].nodes[0].value_before_activation);
        (window as any).net_obj = net;
        setupCanvasContext(draw);
    });
}

function next_netfile() {
    let netn = parseInt(netfile_name.substring("netfile".length, netfile_name.length - ".json".length));
    netfile_name = 'netfile'+(netn+1)+'.json';
    reload_netfile();
}




let netfile_name = 'net.json';
let urlparams = new URLSearchParams(window.location.search);
if (urlparams.has('netfile')) {
    netfile_name = urlparams.get('netfile') + '.json';
}

reload_netfile();







// Interaction

function event_click(e:MouseEvent) {
    let pos = {
        x: e.pageX,
        y: e.pageY
    }

    next_netfile();
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
