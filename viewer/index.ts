
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


let net:Net = {
    activation_fn: (a)=>1,
    derivative_activation_fn: (a)=>0,
    layers: [
        {
            nodes: [
                {
                    bias: 1,
                    input_layer_weights: [],
                    value: 1
                },

                {
                    bias: 1,
                    input_layer_weights: [],
                    value: 1
                },

                {
                    bias: 1,
                    input_layer_weights: [],
                    value: 1
                },
            ]
        },

        {
            nodes: [
                {
                    bias: 1,
                    input_layer_weights: [],
                    value: 1
                },

                {
                    bias: 1,
                    input_layer_weights: [],
                    value: 1
                },

                {
                    bias: 1,
                    input_layer_weights: [],
                    value: 1
                },
            ]
        },

        
        {
            nodes: [
                {
                    bias: 1,
                    input_layer_weights: [],
                    value: 1
                },

                {
                    bias: 1,
                    input_layer_weights: [],
                    value: 1
                },

                {
                    bias: 1,
                    input_layer_weights: [],
                    value: 1
                },
            ]
        },
    ],
    nodes_per_layer: [4,2,2,4]
}


setupCanvasContext(draw);

function draw(c:CanvasRenderingContext2D) {
    let layerN = 0;
    for (let layer of net.layers) {
        let nodeN = 0;
        for (let node of layer.nodes) {
            console.log(nodeN);
            c.strokeStyle = "#ffffff";
            c.lineWidth = 3;
            c.beginPath();
            let layerSpacing = (window.innerWidth) / (1 + net.layers.length);
            let nodeSpacing = (window.innerHeight) / (1 + layer.nodes.length);
            let rad = 25;
            c.ellipse((1+layerN)*layerSpacing, (1+nodeN) * nodeSpacing, 25, 25, 0, 0, 2*Math.PI);
            c.stroke();
            nodeN++;
        }
        layerN++;
    }

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

