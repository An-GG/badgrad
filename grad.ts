
import * as readline from "readline";
import { SIGWINCH } from "constants";
import fs from 'fs';
import { MnistReader } from "./interface";

export type Node = {
    value: number
    bias: number
    input_layer_weights: number[] //first layer wont have obv
}

export type Layer = {
    nodes: Node[]
}

export type LayerValues = Node['value'][];

export type Net = {
    layers: Layer[],
    nodes_per_layer: number[],
    activation_fn: (i:number) => number,
    derivative_activation_fn: (i:number) => number,
    training_metadata?:TrainingMetadata
}


type NetConfig = Omit<Net, 'layers'>
type FromNetfileConfig = Omit<NetConfig, 'nodes_per_layer'>

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
                input_layer_weights: []
            }
            if (layerN > 0) { 
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

function get_net_copy(net:Net):Net {
    return {
        activation_fn: net.activation_fn,
        derivative_activation_fn: net.derivative_activation_fn,
        layers: JSON.parse(JSON.stringify(net.layers)),
        nodes_per_layer: JSON.parse(JSON.stringify(net.layers)),
        training_metadata: JSON.parse(JSON.stringify(net.training_metadata))
    }
}


// TODO assumes input is non-negative ?? or does it work for neg?
function calc_net(net:Net, input:LayerValues, modifyOriginal?:boolean):number[] {
    
    // NodeValue = avn_fn( NodeBias + (for each PNode in PrevLayer: PNode_k.value * Weight_k  /  #_of_Weights) )
    
    let isolated_net = get_net_copy(net);

    // Calc First Layer Vals
    for (let nodeN = 0; nodeN < isolated_net.layers[0].nodes.length; nodeN++) {
        let val = net.activation_fn( isolated_net.layers[0].nodes[nodeN].bias + input[nodeN] );
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

            isolated_net.layers[layerN].nodes[nodeN].value = net.activation_fn( node.bias + weightedSum ); 
            nodeN++;
        }
        layerN++;
    }

    if (modifyOriginal) { net.layers = isolated_net.layers; }    
    return get_net_layer_vals(isolated_net, isolated_net.layers.length - 1);
}

function calc(net:Net, input:LayerValues):number[] {
    if (input) { set_net_input(net, input); }

    let lN = 0;
    for (let layer of net.layers) {
        let nodeN = 0;
        for (let node of layer.nodes) {
            let linkSum = 0;
            let linkN = 0;
            if (node.input_layer_weights != undefined) {
                for (let w of node.input_layer_weights) {
                    linkSum+=(w * net.layers[lN-1].nodes[linkN].value);
                    linkN++;
                }
            } else {
                node.value += node.bias;
                continue;
            }
            
            // contribution needs to be inversely proportional to # of weights, otherwise PD will be unfairly higher
            linkSum = linkSum / linkN; 

            node.value = net.activation_fn( linkSum + node.bias );
            nodeN++;
        }
        lN++;
    }
    return get_net_layer_vals(net, net.layers.length - 1);
}


function average_summed_nudge(nudge: NetNudge, n:number):NetNudge {
    let average_nudge:NetNudge = JSON.parse(JSON.stringify(nudge));    
    let layerN = 0;
    for (let layer of average_nudge) {
        let nodeN = 0;
        for (let l_node of layer.nodeNudges) {
            l_node.biasNudge = l_node.biasNudge / n;
            let wN = 0;
            for (let weight of l_node.weightNudges) {
               l_node.weightNudges[wN] = weight / n;
               wN++;
            }
            nodeN++;
        }
        layerN++;
    }
    return average_nudge;
}


function nudge_network(net:Net, nudge: NetNudge, scalar: number):Net {
    let modded_net:Net = net;
    modded_net.layers = JSON.parse(JSON.stringify(net.layers));
    let layerN = 0;
    for (let layer of nudge) {
        let nodeN = 0;
        for (let l_node of layer.nodeNudges) {
            modded_net.layers[layerN].nodes[nodeN].bias += (l_node.biasNudge * scalar);
            let wN = 0;
            let weights = modded_net.layers[layerN].nodes[nodeN].input_layer_weights;
            if (weights) {    
                for (let weight of l_node.weightNudges) {
                    weights[wN] += l_node.weightNudges[wN] * scalar;
                    wN++;
                }
            }
            nodeN++;
        }
        layerN++;
    }
    return modded_net
}



type TrainingDataBatch = { inputLayer: LayerValues, outputLayer: LayerValues }[];

type TrainingMetadata = {
    error: number,
}





function train(net:Net, training_data: TrainingDataBatch):Net & { training_metadata:TrainingMetadata } {

    // Nudges are all just summed and divided at end
    let nudge: NetNudge = [];

    for (let training_pair of training_data) {
        
        // Strategy
        //
        //

        

    } 
    

    return {}  as any;
}


function train_net(net:Net, trainingData: TrainingDataBatch):Net & { training_metadata:TrainingMetadata } {
    let average_nudge: NetNudge = [];
    // setup nudge to be same as net structure
    let layerN = 0; 
    for (let num_nodes of net.nodes_per_layer) {
        // we can ignore prevLayerPDs because we only need it during backprop
        // TODO: make it optional
        let layernudge: LayerNudge = {
            prevLayerPDs: [],
            nodeNudges: []
        }
        for (let i = 0; i < num_nodes; i++) {
            let nodenudge: NodeNudge = {
                biasNudge: 0,
                weightNudges: []
            }    
            if (layerN > 0) {
                for (let j = 0; j < net.nodes_per_layer[layerN-1]; j++) {
                    nodenudge.weightNudges.push(0);
                }
            }
            layernudge.nodeNudges.push(nodenudge);
        }
        average_nudge.push(layernudge);
        layerN++;
    }
   
    // sum the nudges from each training iteration, avg
    let sumerr = 0;

    let trainingIteration = 0;    
    for (let d of trainingData) {
        calc_net(net, d.inputLayer);
        let netnudge = backprop_net(net, d.outputLayer);
        

        let layerN = 0;
        for (let layer of average_nudge) {
            let nodeN = 0;
            for (let l_node of layer.nodeNudges) {
                l_node.biasNudge += netnudge[layerN].nodeNudges[nodeN].biasNudge;
                let wN = 0;
                for (let weight of l_node.weightNudges) {
                    l_node.weightNudges[wN] += netnudge[layerN].nodeNudges[nodeN].weightNudges[wN];
                    wN++;
                }
                nodeN++;
            }
            layerN++;
        }
        trainingIteration++;
    }

    let avg = average_summed_nudge(average_nudge, trainingIteration);    
    for (let n of avg[avg.length - 1].nodeNudges) {
        sumerr += n.biasNudge*n.biasNudge; 
    }
    let err = Math.sqrt(sumerr);

    let out:Net = nudge_network(net, avg, 0.05);
    out.training_metadata = {
        error: err,
    }
    return out as any;
}



type NodeNudge = {
    biasNudge:number, 
    weightNudges:number[]
};

type LayerNudge = {
    prevLayerPDs:number[],
    nodeNudges:NodeNudge[]
};

type NetNudge = LayerNudge[];

function backprop_net(net: Net, target_output: number[]): LayerNudge[] {
    let netnudges: LayerNudge[] = [];
    let currentPDs: number[] = [];

    let nodeN = 0;
    for (let n of net.layers[net.layers.length-1].nodes) {
       currentPDs.push( 2 * (target_output[nodeN] - n.value));
       nodeN++;
    }

    // all but first layer
    for (let lN = net.layers.length - 1; lN > 0; lN--) {
        let nudge = get_layer_PDs(net.layers[lN], net.layers[lN-1], currentPDs, net);
        netnudges.push(nudge);
        currentPDs = nudge.prevLayerPDs;
    }

    // final (first / input) layer
    netnudges.push(get_layer_PDs(net.layers[0], { nodes:[] }, currentPDs, net));

    netnudges.reverse();
    return netnudges;
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

                // weight partial derivative is divided by num nodes in calculation, so the gradient must be as well
                let nWeights = node.input_layer_weights.length;
                wPD = nWeights == 0 ? 0 : wPD / nWeights;

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





function save_net(net:Net, name:string) {
    let save_obj = {
        layers:net.layers,
        training_metadata: net.training_metadata    
    }
    
    fs.writeFileSync("viewer/netfile"+name+".json", JSON.stringify(save_obj));
}





function relu(x:number):number {
    if (x > 0) { return x; }
    return 0;
}
function derivative_relu(x:number):0|1 {
    if (x > 0) { return 1; }
    return 0;
}

function net_from_netfile(filename:string, cfg:FromNetfileConfig):Net {
    let layers:Layer[] = JSON.parse(fs.readFileSync("viewer/netfile"+filename+".json").toString());
    let nodenums = [];
    for (let l of layers) {
        nodenums.push(l.nodes.length);
    }
    return {
        activation_fn: cfg.activation_fn,
        derivative_activation_fn: cfg.derivative_activation_fn,
        layers: layers,
        nodes_per_layer: nodenums
    }
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

function TRAIN_TEST() {
    
    let newnet = new_net({
        activation_fn: relu,
        derivative_activation_fn: derivative_relu,
        nodes_per_layer: [784, 32, 10, 3]
    }, (i)=>{ return (Math.random() - 0.5); });


    let training = [
            {
                inputLayer: [1,0,0,0,0,0,0],
                outputLayer: [0,1,0]
            },
            {
                inputLayer: [0,0,0,0,0,0,5],
               outputLayer: [2,0,0]
            }
    ];

    calc_net(newnet, training[1].inputLayer);
    save_net(newnet, "1");


    for (let i = 2; i < 1000; i++) {
        newnet = train_net(newnet, training);
        
        console.log(calc_net(newnet, training[1].inputLayer));

        if (i == 999) { calc_net(newnet, training[0].inputLayer); }

        save_net(newnet, i.toString());


    }

}




async function TRAIN_MNIST() {
    
    let mnist_net = new_net({
        activation_fn: relu,
        derivative_activation_fn: derivative_relu,
        nodes_per_layer: [784, 32, 10],
    }, (i)=>{ return (Math.random() - 0.5); });

    let imgreader = await MnistReader.getReader("TRAINING", "IMAGES");
    let lblreader = await MnistReader.getReader("TRAINING", "LABELS");

    let total_iterations = 1000;
    let batch_size = 60;

    let label = lblreader.next();
    let img = imgreader.next();

    save_net(mnist_net, "0");        
    

    for (let i = 1; i < total_iterations; i++) {
        let batch: TrainingDataBatch = [];        

        let npass = 0;
        let nfail = 0;
    
        for (let b = 0; b < batch_size; b++) {

            label = lblreader.next();
            img = imgreader.next();

            let result = calc_net(mnist_net, img);
            let chosen = maxIndex(result);
            if (chosen == label) { npass++; } else { nfail++; }

            let outlayer = [0,0,0,0,0,0,0,0,0,0];
            outlayer[label] = 1;
                    
            batch.push({
                inputLayer: img,
                outputLayer: outlayer
            });
            
        }
        mnist_net = train_net(mnist_net, batch);
        save_net(mnist_net, i.toString());        
        console.log((mnist_net));
        console.log("Iteration: "+i.toString() + " " + (npass/(npass+nfail)) + " " + mnist_net.training_metadata.error);
    }

}

TRAIN_MNIST();

// TODO Output can currently only be positive due to relu, 
//
//  - you could normalize the training data & make negatives positive, and then make the values which are supposed to be negative
//      - this is bad because you loose information
//
//  i think you can do either of the following and i think all are equivalent in effectiveness (but maybe not?)
//  - you can add offset to all training data output to make everything positive, maintaining linear difference between values
//      - idk this sounds annoying, using unsigned ints it should work fine though
//  - you can have whether or not a value is negative be a seperate node (this is like if net was digital, same as sticking binary val into nodes if int is signed)
//      - an extra node is kinda alot (but since its only output layer maybe not that big of a deal?)
//  - you can make the final layer of the net have the activation function y=x
//      - instead of this, i thought abt using final (non-networked) multiplier, like a node with a single previous layer node that does not go through activationfn
//          - this wont work, this is same as making values positive and negating later because net is forced to set multiplier to smth negative
//      - instead, this is basically just a final linear transformation that still has access to all the net's information and the fact that a value is negative can be backpropigated
//
//
// TODO fix issue with large num of weights
//
//  - layers with a lot of weights for each node are summing to really large, so it does not make sense to just allow the net to self adjust because you would need a very small scalar value (current is 0.05) which is not efficient for other layers
//  - need to make contribution proportional to 1 / num_weights 
//  - this will (probably?) affect derivative (wait yes this is good, rn problem is derivative for these is too high) 
