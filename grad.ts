#!/usr/bin/env node
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
    activation_fn: (i:number, pos:{ lN:number, nN:number }, nodes_per_layer:number[]) => number,
    derivative_activation_fn: (i:number, pos:{ lN:number, nN:number }, nodes_per_layer:number[]) => number,
    training_metadata?:TrainingMetadata
}


type NetConfig = Omit<Net, 'layers'>
type FromNetfileConfig = Omit<NetConfig, 'nodes_per_layer'>

type ParameterInitialzerInputs = 
    { paramType: "NodeBias",  layerN:number, nodeN:number, net:Net } |
    { paramType: "Weight",    layerN:number, nodeN:number, weightN:number, net:Net }


function new_net(cfg:NetConfig & { init_fn: (inputs:ParameterInitialzerInputs)=>number }):Net {
    let param_init_function = cfg.init_fn;
    let out:Net = {
        layers: [],
        nodes_per_layer:cfg.nodes_per_layer,
        activation_fn:cfg.activation_fn,
        derivative_activation_fn:cfg.derivative_activation_fn,
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
            newnode.value = cfg.activation_fn(newnode.value_before_activation, { lN:layerN, nN:nodeN }, cfg.nodes_per_layer);
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
    
    
    let isolated_net = get_net_copy(net);

    // Calc First Layer Vals
    for (let nodeN = 0; nodeN < isolated_net.layers[0].nodes.length; nodeN++) {
        isolated_net.layers[0].nodes[nodeN].value_before_activation = isolated_net.layers[0].nodes[nodeN].bias + input[nodeN];
        let val = net.activation_fn(isolated_net.layers[0].nodes[nodeN].value_before_activation, {lN:0,nN:nodeN}, isolated_net.nodes_per_layer);
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
                weightedSum += (w * prevLayer.nodes[wN].value);
                wN++;
            }

            isolated_net.layers[layerN].nodes[nodeN].value_before_activation = node.bias + weightedSum; 
            isolated_net.layers[layerN].nodes[nodeN].value = net.activation_fn( node.bias + weightedSum, {lN:layerN, nN:nodeN}, isolated_net.nodes_per_layer);
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
    rms_error: number,
    avg_error: number
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

// TODO ensure jsonifyability
function cp<T>(a:T):T {
    return JSON.parse(JSON.stringify(a));
}

function train_net(net:Net, training_data: TrainingDataBatch):Net & { training_metadata:TrainingMetadata } {

    let isolated_net:Net = get_net_copy(net);

    let calculated_grads: NetGradient[] = [];
    let sum_scalar_error = 0;
    let sum_avg_error = 0;

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
        let local_scalar_error = 0;
        let local_avg_error = 0;
        for (let nodeN = 0; nodeN < vector_error.length; nodeN++) {
            let diff = vector_error[nodeN];
            local_scalar_error += diff * diff;
            local_avg_error += Math.abs(diff);
            net_grad[currentLayerN][nodeN].nodePD = 2 * diff;
        }
        local_scalar_error = Math.sqrt(local_scalar_error);
        local_avg_error = local_avg_error / vector_error.length;
        sum_scalar_error += local_scalar_error;
        sum_avg_error += local_avg_error;
        
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
                    // * value of current node
                    // * PD of destination node
                    
                    let nodePDSum = 0;
                    let nextLayerNodes = isolated_net.layers[currentLayerN + 1].nodes;

                    let nextLayerNodeN = 0;
                    // TODO optimize
                    while(nextLayerNodeN < nextLayerNodes.length) {
                        
                        // OUTPUT = after activation, INPUT = before
                        
                        // the OUTPUT of this node is multiplied by linkWeight before being added to INPUT of next node, among every other node in this layer
                        let linkWeight = cp( nextLayerNodes[nextLayerNodeN].input_layer_weights[nodeN] )
                        // How useful it would be for the next node's OUTPUT to increase
                        let nextNodePD = cp( net_grad[currentLayerN + 1][nextLayerNodeN].nodePD )
                        // this node's OUTPUT value. affects how much changing the weight of the link to next nodes will impact their INPUT
                        let myNodeValu = cp( node.value )
                        // by how much the next node's OUTPUT will change if the INPUT is changed
                        let nn_dv_avfn = cp( isolated_net.derivative_activation_fn( nextLayerNodes[nextLayerNodeN].value_before_activation, 
                                                                                   { lN: currentLayerN + 1, nN:nextLayerNodeN }, isolated_net.nodes_per_layer ) )
                        // The number of other nodes that will be added to the INPUT of next node (shouldnt need this)
                        // let numWeights = cp( nextLayerNodes[nextLayerNodeN].input_layer_weights.length )

                        // A partial component of this link's weight PD
                        // how useful it would be to change this link's weight 
                        let weightPD_comp = nn_dv_avfn * nextNodePD * myNodeValu;
                        net_grad[currentLayerN + 1][nextLayerNodeN].weightsPD[nodeN] += cp( weightPD_comp )

                        // how useful it would be to change the OUTPUT of this node
                        node_grad.nodePD += cp( nn_dv_avfn * nextNodePD * linkWeight )
                        nextLayerNodeN++;
                    }
                }

                // by how much this node's OUTPUT will change if the INPUT is changed
                let my_dv_avfn = cp( isolated_net.derivative_activation_fn( node.value_before_activation, { lN: currentLayerN, nN:nodeN }, isolated_net.nodes_per_layer ) )
                
                // Calculate Bias Grad for each node in this layer
                node_grad.biasPD = cp( node_grad.nodePD * my_dv_avfn);

            }
            currentLayerN--;
        }
        calculated_grads.push(net_grad);
    }

    let rms_error = sum_scalar_error / training_data.length;
    let avg_error = sum_avg_error / training_data.length;

    let avg = average_grads(calculated_grads);
    let learnRate = parseFloat(getArgs().learnRate);
    let applied_grad_net = apply_gradient(net, avg, learnRate) as Net & { training_metadata:TrainingMetadata };    
    applied_grad_net.training_metadata = {
        rms_error: rms_error,
        avg_error: avg_error
    } 
    return applied_grad_net;
}


type Netfile = {
    iterations: {[n:string]:{ layers: Net['layers'], training_metadata: Net['training_metadata'] } }
}
function save_net(net:Net, name:string, netfile?:Netfile) {
    let save_obj = {
        layers:net.layers,
        training_metadata: net.training_metadata,
    };
    if (netfile) {
        netfile.iterations[name] = save_obj;
        fs.writeFileSync("viewer/netfile.json", JSON.stringify(netfile));
    } else {
        if (!fs.existsSync("temp_netfiles")) { fs.mkdirSync("temp_netfiles"); }
        fs.writeFileSync("temp_netfiles/"+name+".json", JSON.stringify(net));
    }
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


let args_obj = {
    startFrom: (false as string | false),
    saveEveryNth: ("32" as string),
    learnRate: ("0.01" as string),
    untilRMSError: ("0.001" as string),
    batchSize: ("50" as string),
    batchOrder: ("sequential (choose 'random' or 'sequential')" as 'random' | 'sequential')
} as const;

let supportedArgs = Object.keys(args_obj) as (keyof typeof args_obj)[];
type TrainingArgs = typeof args_obj; 

let supportedArgsStr = "Supported Args:\n";
for (let k in args_obj) {
    supportedArgsStr+= k.padStart(15) + '  default_value=' + args_obj[k as keyof TrainingArgs] + '\n';
}

export type DeepWritable<T> = { -readonly [P in keyof T]: DeepWritable<T[P]> };

function getArgs(): TrainingArgs {
    let out:DeepWritable<TrainingArgs> = JSON.parse(JSON.stringify(args_obj));
    

    for (let a of process.argv.splice(2)) {
        // Verify correctness
        let supported = false; 
        let currentArg:string="";
        for (let sp of supportedArgs) {
            let spArg = '--'+sp+'=';
            if (a.startsWith(spArg)) { supported = true; currentArg=spArg; }
        }

        if (!supported) { throw new Error('Unsupported Argument: '+a+'\n\n'+supportedArgsStr); }

        (out as any)[a.substring(2, currentArg.length-1)] = a.substring(currentArg.length);
        
    }
    
    // No whitespace in args, take first element of whitespace
    for (let aN in out) {
        let argN = aN as (keyof typeof out);
        let val = out[argN];
        if (typeof val == 'string') {
            if (argN == 'batchOrder') {
                out[argN] = val.startsWith('sequential') ? 'sequential' : 'random';
            } else {
                out[argN] = val.split(' ')[0];
            }
        }
    }
                                        
    
    return out;
}


function layer_vector_diff(l:Layer, target:number[], absolute?:boolean): number[] {
    let out =[];
    let i = 0;
    for (let n of l.nodes) {
        let diff = n.value - target[i];
        if (absolute) { diff = Math.abs(diff); }
        out.push(diff);  
        i++;
    }
    return out;
}

function avg_vector_diff(v:number[]):number {
    let s = 0;
    for (let a of v) { s+=a; }
    return s / v.length; 
}

function TRAIN_TEST() {
   
    if (fs.existsSync('temp_netfiles')) { fs.rmSync('temp_netfiles', { recursive: true }); }
    if (fs.existsSync('viewer/netfile.json')) { fs.rmSync('viewer/netfile.json'); }

    let args = getArgs();

    let newnet = new_net({
        activation_fn: relu,
        derivative_activation_fn: derivative_relu,
        nodes_per_layer:  [7, 4, 4, 3],
        init_fn: (i:ParameterInitialzerInputs)=>{ 
            if (i.paramType == 'NodeBias') {
                return 0;   
            } else {
                let n_nodes = i.net.layers[0].nodes.length;
                let kaimingInit = (Math.random())*Math.sqrt(2 / n_nodes);
                
                return kaimingInit;
            }
        }
    });
    
    if (args.startFrom) {
        let n:Net = JSON.parse(fs.readFileSync(args.startFrom).toString());
        for (let key in n) {
            (newnet as any)[key] = (n as any)[key];
        }
    }      


    let training = [
            {
                inputLayer: [1,0,0,0,0,0,0],
                outputLayer: [0,1,0]
            },
            {
                inputLayer: [0,0,0,0,0,0,5],
               outputLayer: [2,0,0]
            },
            {
                inputLayer: [0,2,0,0,0,0,0],
               outputLayer: [3,0,1]
            },
            {
                inputLayer: [0,0,0,0,0,0,2],
               outputLayer: [0,1,6]
            },
    ];

    calc_net(newnet, training[0].inputLayer, true);
    save_net(newnet, "0");
    
    let netfile:Netfile = { iterations: {} };
    let err = 0;
    let n = 0;
    let t0 = (new Date()).getTime();
    let nth_save = 0;
    let prev_err = 0;


    while (true) {
        newnet = train_net(newnet, training);
        err = parseFloat((newnet.training_metadata as any).rms_error);
        if (n % parseInt(args.saveEveryNth) == 0) {
            calc_net(newnet, training[nth_save % training.length].inputLayer, true);
            
            console.log(n.toString().padStart(10, "0")+"   "+
                        (newnet.training_metadata as any).rms_error.toString().padEnd(24, "0")+"    "+
                        Math.abs(prev_err -  err).toString().padEnd(24, "0")+" "+
                        (Math.sign(prev_err - err) == 1 ? '+' : '-') 
            );

            save_net(newnet, (n+1).toString());
            prev_err = JSON.parse(JSON.stringify(err));
            nth_save++;
        }
        if (err < 0.001) {
            break;
        }
        n++;    
    }
    let total_time = (new Date()).getTime() - t0;

    console.log(":::::MARK:::::"); 
    console.log("LAST ITERATION: "+n);
    console.log("FINAL ERROR:    "+err);
    console.log("TOTAL TIME:     "+total_time);
}

function TRAIN_MNIST() {

    // Timer Start
    let t0 = (new Date()).getTime();

    if (fs.existsSync('temp_netfiles')) { fs.rmSync('temp_netfiles', { recursive: true }); }
    if (fs.existsSync('viewer/netfile.json')) { fs.rmSync('viewer/netfile.json'); }

    let args = getArgs();
    
    let newnet = new_net({
        activation_fn: relu,
        derivative_activation_fn: derivative_relu,
        nodes_per_layer:  [784, 32, 10],
        init_fn: (i:ParameterInitialzerInputs)=>{ 
            if (i.paramType == 'NodeBias') {
                return 0;   
            } else {
                if (i.layerN < 1) { return 0; } else {
                    return ((Math.random() - 0.5) * 2) / (i.net.nodes_per_layer[i.layerN] * i.net.nodes_per_layer[i.layerN - 1]);
                }
            }
        }
    });
    
    if (args.startFrom) {
        let n:Net = JSON.parse(fs.readFileSync(args.startFrom).toString());
        for (let key in n) {
            (newnet as any)[key] = (n as any)[key];
        }
    }

    let img_reader = new MnistReader("TRAINING", "IMAGES");
    let lbl_reader = new MnistReader("TRAINING", "LABELS");    
   
    calc_net(newnet, img_reader.next(), true); 
    save_net(newnet, "0");
   
    function get_nth_databatch(n:number): TrainingDataBatch {
        let out:TrainingDataBatch = [];
        
        for (let i = 0; i < parseInt(args.batchSize); i++) {
            let pos:number;
            if (args.batchOrder == 'sequential') {
                pos = n * parseInt(args.batchSize);
            } else {
                pos = Math.random() * (lbl_reader.length - 1);
            }
            img_reader.setHeadPosition(pos);
            lbl_reader.setHeadPosition(pos);
            let outlayer = [0,0,0,0,0,0,0,0,0,0];
            outlayer[lbl_reader.next()] = 1;
            out.push({ inputLayer: img_reader.next(), outputLayer: outlayer });
        }

        return out;
    }

    let batchN = 0;
    let nth_save = 0;
    let err = 0;
    let prev_saved_err = 0;
    
    while (true) {
        let batch = get_nth_databatch(batchN);
        newnet = train_net(newnet, batch);
        if (batchN % parseInt(args.saveEveryNth) == 0) {
            calc_net(newnet, batch[nth_save % batch.length].inputLayer, true);

            let log = 
                    batchN.toString().padStart(10, "0") +
                    "    " +
                    (newnet.training_metadata as any).rms_error.toString().padEnd(24, "0") +
                    "    " +
                    Math.abs(prev_saved_err - err).toString().padEnd(24, "0") +
                    "    " +
                    (Math.sign(prev_saved_err - err) == 1 ? '+' : '-')
                    ;

            save_net(newnet, (batchN + 1).toString());
            prev_saved_err = cp( err );
            nth_save++;
            console.log(log);
        }
        if (err < parseInt(args.untilRMSError)) { break; }
        batchN++;
    }

    let total_time = (new Date()).getTime() - t0;

    console.log(":::::MARK:::::"); 
    console.log("LAST ITERATION: "+batchN);
    console.log("FINAL ERROR:    "+err);
    console.log("TOTAL TIME:     "+total_time);

}



TRAIN_MNIST();
//TRAIN_TEST();
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



/**
 We initialize weight with a normal distribution with mean 0 and variance std, and the ideal distribution of weight after relu should have slightly incremented mean layer by layer and variance close to 1. We can see the output is close to what we expected. The mean increment slowly and std is close to 1 in the feedforward phase. And such stability will avoid the vanishing gradient problem and exploding gradient problem in the backpropagation phase.
**/
