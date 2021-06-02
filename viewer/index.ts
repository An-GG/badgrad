
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
        for (let node of layer.nodes) {
            console.log(nodeN);
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
                c.beginPath();
                c.fillStyle = "#000000";
                c.ellipse(point.x, point.y, 25, 25, 0, 0, 2*Math.PI);
                c.fill();
                c.stroke();
            }

            
            if (node.input_layer_weights && node.input_layer_weights.length > 0 && mode == "LINE_ONLY") {
                let wN = 0;
                let weightSpacing = (window.innerHeight) / (1 + node.input_layer_weights.length);
                for (let w of node.input_layer_weights) {
                    let prevpoint = {
                        x: layerN * layerSpacing,
                        y: (1+wN) * weightSpacing
                    }
                    c.beginPath();
                    c.strokeStyle = "#ffffff";
                    c.lineWidth = Math.abs((sigmoid(w)) - 0.5) * 10;
                    console.log(sigmoid(w));
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







loadJSON('net.json', (r) => {
    net = JSON.parse(r);
    setupCanvasContext(draw);
});



