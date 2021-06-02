import { MnistReader } from "./interface";
import * as readline from "readline";
import { SIGWINCH } from "constants";

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
        out.layers[layerN].nodes = newlayer;
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


















