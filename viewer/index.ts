import { Net } from '../net'


let net:Net = {
    activation_fn: (a)=>1,
    derivative_activation_fn: (a)=>0,
    layers: [],
    nodes_per_layer: [4,2,2,4]
}


setupCanvasContext(draw);

function draw(c:CanvasRenderingContext2D) {
    
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

