import { MnistReader } from "./interface";
import * as readline from "readline";
import { SIGWINCH } from "constants";
import fs from 'fs';

export type Node = {
    value: number
    bias: number
    input_layer_weights?: number[] //first layer wont have obv
}

export type Layer = {
    nodes: Node[]
}

export type LayerValues = Node['value'][];

export type Net = {
    layers: Layer[],
    nodes_per_layer: number[],
    activation_fn: (i:number) => number,
    derivative_activation_fn: (i:number) => number
}


type NetConfig = Omit<Net, 'layers'>

type ParameterInitialzerInputs = 
    { paramType: "NodeValue", layerN:number, nodeN:number, net:Net } |
    { paramType: "NodeBias",  layerN:number, nodeN:number, net:Net } |
    { paramType: "Weight",    layerN:number, nodeN:number, weightN:number, net:Net }


function new_net(cfg:NetConfig, param_init_function:(inputs:ParameterInitialzerInputs)=>number):Net {
    let out:Net = {
        layers: [],
        nodes_per_layer:cfg.nodes_per_layer,
        activation_fn:cfg.activation_fn,
        derivative_activation_fn:cfg.derivative_activation_fn 
    }
    let layerN = 0;
    for (let numnodes of cfg.nodes_per_layer) {
        let newlayer:Node[] = [];
        for (let nodeN = 0; nodeN < numnodes; nodeN++) {
            let newnode:Node = {
                bias: param_init_function({ paramType:"NodeBias", layerN:layerN, nodeN:nodeN, net:out }),
                value: param_init_function({ paramType:"NodeValue", layerN:layerN, nodeN:nodeN, net:out }),
            }
            if (layerN > 0) { 
                newnode.input_layer_weights = [];
                for (let weightN = 0; weightN < cfg.nodes_per_layer[layerN - 1]; weightN++) {
                    newnode.input_layer_weights.push(
                        param_init_function({ paramType:"Weight", layerN:layerN, nodeN:nodeN, weightN:weightN, net:out })                                
                    );
                }
            }
            newlayer.push(newnode);
        }
        out.layers.push({
            nodes: newlayer
        });

        layerN++;
    } 
    return out;
}


function set_net_input(net:Net, setTo:LayerValues) {
    let i = 0;
    for (let n of net.layers[0].nodes) {
        n.value = setTo[i];
        i++;
    }
}

function get_net_layer_vals(net:Net, layerN:number):LayerValues {
    let out:LayerValues = [];
    for (let n of net.layers[layerN].nodes) {
        out.push(n.value);
    }
    return out;
}

function calc_net(net:Net, input?:LayerValues):number[] {
    if (input) { set_net_input(net, input); }

    let lN = 0;
    for (let layer of net.layers) {
        for (let node of layer.nodes) {
            if (node.input_layer_weights == undefined) { continue; }
            
            let linkSum = 0;
            
            let linkN = 0;
            for (let w of node.input_layer_weights) {
                linkSum+=(w * net.layers[lN-1].nodes[linkN].value);
                linkN++;
            }
            
            node.value = net.activation_fn( linkSum + node.bias );
        }
        lN++;
    }
    return get_net_layer_vals(net, net.layers.length - 1);
}




function train_net(net:Net, trainingData: { inputLayer: Layer, outputLayer: Layer }[]) {

    
}

type NodeNudge = {
    biasNudge:number, 
    weightNudges:number[]
}

type LayerNudge = {
    prevLayerPDs:number[],
    nodeNudges:NodeNudge[]
}

function backprop_net(net: Net, target_output: number[]) {
    let netnudges: LayerNudge[] = [];
    let currentPDs: number[] = [];

    let nodeN = 0;
    for (let n of net.layers[net.layers.length-1].nodes) {
       currentPDs.push( 2 * (target_output[nodeN] - n.value));
       nodeN++;
    }

    for (let lN = net.layers.length - 1; lN > 0; lN--) {
        let nudge = get_layer_PDs(net.layers[lN], net.layers[lN-1], currentPDs, net);
        netnudges.push(nudge);
        currentPDs = nudge.prevLayerPDs;
    }

    netnudges.reverse();
}

function get_layer_PDs(layer:Layer, prev:Layer, thislayerPDs:number[], net:Net): LayerNudge {
    let out:LayerNudge = {
        prevLayerPDs:[],
        nodeNudges:[]
    }
    let prevLayerPDTotals:number[][] = []
    for (let n of prev.nodes) { prevLayerPDTotals.push([]); } 

    let nodeN = 0;
    for (let node of layer.nodes) {
        let nodeNudge: NodeNudge = {
            biasNudge: thislayerPDs[nodeN],
            weightNudges:[]
        }
        
        let wN = 0;
        if (node.input_layer_weights) {
            for (let w of node.input_layer_weights) {
                
                // Weight Partial Derivative = 
                // (InputNode Value) * (DerivativeActivation(CurrentValue)) * PDCurrentValue
                
                let wPD = prev.nodes[wN].value * thislayerPDs[nodeN] * net.derivative_activation_fn(node.value);
                // The relu derivative will work bc we're lucky, but need to fix TODO
                // node.value is output of activation_fn(), cannot plug into derivative_fn

                let prevValuePD = w * thislayerPDs[nodeN] * net.derivative_activation_fn(node.value);

                nodeNudge.weightNudges.push(wPD);
                prevLayerPDTotals[wN].push(prevValuePD);
                wN++;
            }
        }
        out.nodeNudges.push(nodeNudge);
        nodeN++;
    }

    for (let pdarr of prevLayerPDTotals) {
        let sum = 0;
        for (let a of pdarr) {
            sum+=a;
        }
        out.prevLayerPDs.push( sum / pdarr.length );
    }

    return out;
}





function save_net(net:Net) {
    let save_obj = {
        layers:net.layers
    }
    fs.writeFileSync("viewer/net.json", JSON.stringify(save_obj));
}





function relu(x:number):number {
    if (x > 0) { return x; }
    return 0;
}
function derivative_relu(x:number):0|1 {
    if (x > 0) { return 1; }
    return 0;
}




let newnet = new_net({
    activation_fn: relu,
    derivative_activation_fn: derivative_relu,
    nodes_per_layer: [7, 4, 4, 3]
}, (i)=>{ return Math.random(); });

save_net(newnet);










