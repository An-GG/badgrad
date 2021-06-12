
type Neuron = {
    value: number
    bias: number
    input_layer_weights?: number[] //first layer wont have obv
}

type Layer = {
    nodes: Neuron[]
}

type LayerValues = Neuron['value'][];

type Net = {
    layers: Layer[],
    nodes_per_layer: number[],
    activation_fn: (i:number) => number,
    derivative_activation_fn: (i:number) => number
}


let net:Net;



function sigmoid(x:number):number {
    return (1 / (1 + Math.pow(Math.E, -x)));
}

function drawNet(c:CanvasRenderingContext2D, mode: "LINE_ONLY" | "NODE_ONLY") {
    
    let layerN = 0;
    for (let layer of net.layers) {
        let nodeN = 0;

        let maxbias = 1;
        let maxval = 1;
        for (let n of layer.nodes) {
            //if (Math.abs(n.bias) > maxbias) { maxbias = Math.abs(n.bias) };
            //if (Math.abs(n.value) > maxval) { maxval = Math.abs(n.value) };
        } 

        for (let node of layer.nodes) {
            c.strokeStyle = "#ffffff";
            c.lineWidth = 3;
            
            let layerSpacing = (window.innerWidth) / (1 + net.layers.length);
            let nodeSpacing = (window.innerHeight) / (1 + layer.nodes.length);
            let rad = 25;
            let point = {
                x: (1+layerN) * layerSpacing,
                y: (1+nodeN) * nodeSpacing
            }
            if (mode == "NODE_ONLY") {


                c.lineWidth = 3;
                c.beginPath();
                let nodebias_rel = Math.round(Math.abs(node.bias*255)); //parseInt((sigmoid(Math.abs(node.bias) / maxbias)*255).toString());
                let nodeval_rel = Math.round(Math.abs(node.value*255));// parseInt((sigmoid(Math.abs(node.value) / maxval)*255).toString());
                
                if (nodebias_rel > 255) { nodebias_rel = 255; }
                if (nodeval_rel > 255) { nodeval_rel = 255; }

                console.log("===========");
                console.log(nodeval_rel);
                if (node.bias > 0) {
                    c.strokeStyle = "#0000"+(nodebias_rel.toString(16).padStart(2,"0"))
                } else {
                    c.strokeStyle = "#"+(nodebias_rel.toString(16).padStart(2,"0")) + "0000";
                }

                if (node.value > 0) {
                    c.fillStyle = "#0000"+(nodeval_rel.toString(16).padStart(2,"0"));
                } else {
                    c.fillStyle = "#"+(nodeval_rel.toString(16).padStart(2,"0")) + "0000";
                }

                console.log(node);
                console.log(c.fillStyle);
                c.ellipse(point.x, point.y, 25, 25, 0, 0, 2*Math.PI);
                c.fill();
                c.stroke();
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
                        c.strokeStyle = "#0000"+weightval_rel.toString(16).padStart(2,"0");
                    } else {
                        c.strokeStyle = "#"+weightval_rel.toString(16).padStart(2,"0")+"0000";
                    }
                    c.lineWidth = Math.abs((sigmoid(w)) - 0.5) * 10;
                    c.moveTo(point.x, point.y);
                    c.lineTo(prevpoint.x, prevpoint.y);
                    c.stroke();
                    wN++;
                }
            }
            nodeN++;
        }
        layerN++;
    }

}



function draw(c:CanvasRenderingContext2D) {
    drawNet(c, "LINE_ONLY");
    drawNet(c, "NODE_ONLY");
}

function setupCanvasContext(drawFunc:(ctx:CanvasRenderingContext2D)=>any) {
    let canvas = document.getElementById("canvas") as HTMLCanvasElement;
    let context = canvas.getContext('2d')!;
    function onResize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        context.clearRect(0, 0, canvas.width, canvas.height);
        drawFunc(context); 
    }
    window.addEventListener('resize', onResize);
    onResize();
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





let netfile_name = 'net.json';
let urlparams = new URLSearchParams(window.location.search);
if (urlparams.has('netfile')) {
    netfile_name = urlparams.get('netfile') + '.json';
}

loadJSON(netfile_name, (r) => {
    net = JSON.parse(r);
    setupCanvasContext(draw);
});



