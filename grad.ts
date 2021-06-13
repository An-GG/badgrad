
import * as readline from "readline";
import { SIGWINCH } from "constants";
import fs from 'fs';
import { MnistReader } from "./interface";

export type Node = {
    value_before_activation: number
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
                value_before_activation: 0,
                input_layer_weights: [],
                value:0
            }
            newnode.value = cfg.activation_fn(newnode.value_before_activation);
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
        training_metadata: net.training_metadata ? JSON.parse(JSON.stringify(net.training_metadata)) : undefined
    }
}


// TODO assumes input is non-negative ?? or does it work for neg?
function calc_net(net:Net, input:LayerValues, modifyOriginal?:boolean):number[] {
    
    // NodeValue = avn_fn( NodeBias + (for each PNode in PrevLayer: PNode_k.value * Weight_k  /  #_of_Weights) )
    
    let isolated_net = get_net_copy(net);

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
    return get_net_layer_vals(isolated_net, isolated_net.layers.length - 1);
}

function apply_gradient(net:Net, grad:NetGradient, learnRate:number):Net {
    let isolated_net = get_net_copy(net);
    
    let ln = 0;
    for (let layer of isolated_net.layers) {
        let nn = 0;
        for (let node of layer.nodes) {
            node.bias += grad[ln][nn].biasPD * learnRate;
            let wn = 0;
            for (let w of node.input_layer_weights) {
                node.input_layer_weights[wn] += grad[ln][nn].weightsPD[wn] * learnRate;
                wn++;
            }
            nn++;
        }
        ln++;
    }

    return isolated_net;
}


type TrainingDataBatch = { inputLayer: LayerValues, outputLayer: LayerValues }[];

type TrainingMetadata = {
    error: number,
}

type NodeGradient = {
    // The gradient of the current node, for ex. last layer nodes, nodePD = 2 * difference
    nodePD:number,
    // The gradient / direction & magnitude by which to change each input weight (length = # of nodes in prev layer. for first layer this = [])
    weightsPD:number[],
    // The gradient / direction & magnitude by which to change the node's bias
    biasPD:number
}
type LayerGradient = NodeGradient[];
type NetGradient = LayerGradient[];


// Initializes an empty gradient with zeros for a givent net structure
function get_blank_gradient(net:Net):NetGradient {
    let g:NetGradient = [];
    for (let l of net.layers) {
        let gL:LayerGradient = [];
        for (let n of l.nodes) {
            
            let weightsPD = [];
            for (let w of n.input_layer_weights) { weightsPD.push(0); }

            let gN:NodeGradient = {
                nodePD:0,
                weightsPD:weightsPD,
                biasPD:0
            }
            gL.push(gN);
        }
        g.push(gL);
    }
    return g;
}

function average_grads(grads:NetGradient[]):NetGradient {
    let avg:NetGradient = JSON.parse(JSON.stringify(grads[0]));
    let sum:NetGradient = JSON.parse(JSON.stringify(grads[0]));
    for (let gradN = 1; gradN < grads.length; gradN++) {
        let grad = grads[gradN];
        for (let ln = 0; ln < grad.length; ln++) {
            let layer = grad[ln];
            for (let nn = 0; nn < layer.length; nn++) {
                let node_grad = layer[nn];
                sum[ln][nn].biasPD += node_grad.biasPD;   
                avg[ln][nn].biasPD = sum[ln][nn].biasPD / (gradN + 1);
                sum[ln][nn].nodePD += node_grad.nodePD;
                avg[ln][nn].nodePD = sum[ln][nn].nodePD / (gradN + 1);
                for (let wN = 0; wN < node_grad.weightsPD.length; wN++) {
                    sum[ln][nn].weightsPD[wN] += node_grad.weightsPD[wN];
                    avg[ln][nn].weightsPD[wN] = sum[ln][nn].weightsPD[wN] / (gradN + 1);
                }
            } 
        }
    }
    return avg;
}

function train_net(net:Net, training_data: TrainingDataBatch):Net & { training_metadata:TrainingMetadata } {

    let isolated_net:Net = get_net_copy(net);

    let calculated_grads: NetGradient[] = [];
    let sum_scalar_error = 0;


    for (let training_pair of training_data) {
        
        let net_grad: NetGradient = get_blank_gradient(isolated_net);
        
        // Calculate net fully 
        calc_net(isolated_net, training_pair.inputLayer, true);
        let vector_error = [];
        for (let nodeN = 0; nodeN < training_pair.outputLayer.length; nodeN++) {
            vector_error.push( training_pair.outputLayer[nodeN] - isolated_net.layers[net.layers.length - 1].nodes[nodeN].value );
        }

        // Start at last layer and get loss (PD) for each node in last layer
        let currentLayerN = net.layers.length - 1;        

        // For final layer, PD is 2 * (difference)
        let scalar_error = 0;
        for (let nodeN = 0; nodeN < vector_error.length; nodeN++) {
            let diff = vector_error[nodeN];
            scalar_error += diff * diff;
            net_grad[currentLayerN][nodeN].nodePD = 2 * diff;
        }
        scalar_error = Math.sqrt(scalar_error);
        sum_scalar_error += scalar_error;
        
        // Now, we can use PD to calculate previous layer PDs recursively
        while (currentLayerN >= 0) {
            
            // For each node in layer, calc NodeGradient

            for (let nodeN = 0; nodeN < isolated_net.layers[currentLayerN].nodes.length; nodeN++) {
                let node = isolated_net.layers[currentLayerN].nodes[nodeN];
                let node_grad = net_grad[currentLayerN][nodeN];

                // For final layer, PD is already defined
                if (currentLayerN < net.layers.length - 1) {
                    // To calculate PD, need to average this EQ across all nodes for which this node in an input, 
                    // (Link_Weight / # Nodes in this Layer) 
                    // * derivative_activation( Value before activation of destination node ) 
                    // * PD of destination node
                    
                    let nodePDSum = 0;
                    let nextLayerNodes = isolated_net.layers[currentLayerN + 1].nodes;

                    let nextLayerNodeN = 0;
                    while(nextLayerNodeN < nextLayerNodes.length) {
                        let linkWeight = nextLayerNodes[nextLayerNodeN].input_layer_weights[nodeN];
                        let numWeights = nextLayerNodes[nextLayerNodeN].input_layer_weights.length;
                        let nextNodePD = net_grad[currentLayerN + 1][nextLayerNodeN].nodePD;
                        let deriv_avfn = isolated_net.derivative_activation_fn( nextLayerNodes[nextLayerNodeN].value_before_activation );

                        nodePDSum += (linkWeight / numWeights) * deriv_avfn * nextNodePD;
                        
                        nextLayerNodeN++;
                    }
                    node_grad.nodePD = (nodePDSum / nextLayerNodeN);
                }

                // Calculate Bias Grad for each node in this layer
                node_grad.biasPD = node_grad.nodePD * isolated_net.derivative_activation_fn( node.value_before_activation );

                // Calculate Weight Grad
                for (let wN = 0; wN < node.input_layer_weights.length; wN++) {
                    let num_weights = node.input_layer_weights.length;
                    let wgrad = (isolated_net.layers[currentLayerN - 1].nodes[wN].value / num_weights) 
                                * isolated_net.derivative_activation_fn( node.value_before_activation ) 
                                * node_grad.nodePD;
                    node_grad.weightsPD[wN] = wgrad;
                }
            }
            currentLayerN--;
        }
        calculated_grads.push(net_grad);
    }

    let avg_error = sum_scalar_error / training_data.length;

    let avg = average_grads(calculated_grads);
    let applied_grad_net = apply_gradient(net, avg, 0.01) as Net & { training_metadata:TrainingMetadata };    
    applied_grad_net.training_metadata = {
        error: avg_error
    } 
    return applied_grad_net;
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
        nodes_per_layer: [7, 4, 4, 3]
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

    calc_net(newnet, training[1].inputLayer, true);
    save_net(newnet, "1");


    for (let i = 2; i < 1000; i++) {
        newnet = train_net(newnet, training);
        
        console.log(i+"   "+(newnet.training_metadata as any).error);

        calc_net(newnet, training[1].inputLayer, true);
        if (i == 999) { calc_net(newnet, training[0].inputLayer, true); }

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
        console.log("Iteration: "+i.toString() + " " + (mnist_net as any).training_metadata.error) + " " + (npass/(npass+nfail));
    }

}

TRAIN_TEST();

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
