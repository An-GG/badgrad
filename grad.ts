import { MnistReader } from "./interface";
import * as readline from "readline";
import { stringify } from "flatted";

let K_LAYERS = [3, 6, 6, 3];

type link = {
    in:node
    out:node
    weight:number
}
type node = {
    value:number
    in_links:link[]
    out_links:link[]
}
type net = layer[];
type layer = node[];

let net:net = [];


// Build Net
for (let layerNk in K_LAYERS) {
    let layerN = parseInt(layerNk);
    let n_nodes = K_LAYERS[layerN];
    let l:layer = [];
    for (let i=0; i<n_nodes; i++) {
        let n:node = {
            in_links:[],
            out_links:[],
            value:1 // inital value of nodes doesn't matter
        }

        if (layerN > 0) {
            // Add links to prev layer
            for (let prev_node of net[layerN-1]) {
                let newlink:link = {
                    weight: 1,
                    in: prev_node,
                    out: n
                }
                prev_node.out_links.push(newlink);
                n.in_links.push(newlink);
            }
        }

        l.push(n);
    }
    net.push(l);
}


async function train() {
    let imageReader = await MnistReader.getReader("TRAINING", "IMAGES");
    let labelReader = await MnistReader.getReader("TRAINING", "LABELS");

    // Train
    let input = imageReader.next();
    let output = labelReader.next();

    


    let set: {i:number[],o:number[]}[] = [
        { i:[1,0,0.4],o:[0,1,1] },
        { i:[1,0.5,0.5],o:[1,0,0] },
        { i:[0,0,1],o:[0,0,1] },
        { i:[0,0.2,0.4],o:[1,1,1] },
        { i:[1,1,1],o:[0,0,0] },
    ]

    setlayer(set[0].i, net, 0);
    calcnet(net);
    console.log(net_toStr(net));

    let rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });
    let chosen = 0;
    for await (const line of rl) {
        let inp = line.split(' ')[0];
        if (inp[0] == 'l') {
            let outlinks = [];
            let inlinks  = [];
            let layN = parseInt(inp[1]);
            let nodeN = parseInt(inp[2]);
            for (let link of net[layN][nodeN].in_links) {
                inlinks.push(link.weight);
            }
            for (let link of net[layN][nodeN].out_links) {
                outlinks.push(link.weight);
            }
            console.log("Outlinks")
            console.log(outlinks);
            console.log("Inlinks")
            console.log(inlinks)
            continue;
        }

        if (inp[0] == 'j') {
            let i=0;
            for (let lay of net) {
                i++;
                console.log("=============LAYER="+i+"===============")
                for (let node of lay) {
                    console.log("   NODE:  "+node.value.toPrecision(4));
                    let a =     "       INLINKS: ";
                    for (let link of node.in_links) {
                        a+=link.weight.toPrecision(4) + " ";
                    }
                    a+="\n      OUTLINKS: ";
                    for (let link of node.out_links) {
                        a+=link.weight.toPrecision(4) + " ";
                    }
                    console.log(a);
                }
            }
            continue;
        }

        tweak(net, set[chosen].o);
        chosen = Math.floor(Math.random() * 5);
        setlayer(set[chosen].i, net, 0);
        calcnet(net);
        console.log(net_toStr(net));
    }

}


  

train();

// Returns output vector
function calcnet(n:net):layer {
    for (let layer of n) {
        for (let node of layer) {
            if (node.in_links.length == 0) { continue; }
            let sum = 0;
            for (let inlink of node.in_links) {
                sum += inlink.weight * inlink.in.value
            }
            node.value = sum / node.in_links.length;
        }
    }
    return n[n.length-1];
}

let ADJ_K = 1
function tweak(net:net, target:number[]) {
    let currentDiff = [];
    for (let nodeN=0; nodeN<net[net.length-1].length; nodeN++) {
        currentDiff.push(target[nodeN] - net[net.length-1][nodeN].value);
    }

    for (let layerN=net.length-1; layerN>=0; layerN--) {
        let layer = net[layerN];

        let next_layer_helpfulness:number[] = (new Array(layer[0].in_links.length)).fill(0);

        for (let nodeN=0; nodeN<layer.length; nodeN++) {
            let node = layer[nodeN];
            let diff = currentDiff[nodeN];

            // Strengthen the links that pointed in right direction, proportional to how off you are
            for (let linkN=0; linkN<node.in_links.length; linkN++) {
                let link = node.in_links[linkN];
                let contribution = (link.weight * link.in.value) / node.in_links.length;
                let helpfulness = contribution * diff;
                let adj = helpfulness * Math.sign(link.weight) * ADJ_K;
                link.weight = link.weight + adj;
                next_layer_helpfulness[linkN] += helpfulness * Math.sign(link.in.value);
            }
        }
        currentDiff = next_layer_helpfulness;
    }
}

function setlayer(v:number[], net:net, layerN:number) {
    for (let i = 0; i<net[layerN].length; i++) {
        net[layerN][i].value = v[i];
    }
}



function net_toStr(n:net):string {
    let maxH = Math.max(...K_LAYERS);
    let outvalset: string[][] = [];// list of columns, column is maxH and width is same as net
    let shiftHalfUpLayer: boolean[] = [];
    for (let layer of n) {
        let newlayer = [];
        let topOffset = (maxH - layer.length) / 2;
        shiftHalfUpLayer.push((topOffset*2)%2 == 1);
        while (newlayer.length < topOffset) { newlayer.push("       "); }
        for (let node of layer) { newlayer.push("  "+node.value.toPrecision(2)+"  "); }
        while (newlayer.length < maxH) { newlayer.push("       "); }
        outvalset.push(newlayer);
    }
    let out = "";
    for (let i=0; i<maxH;i++) {
        // Print shifted
        for (let layerN=0; layerN<outvalset.length; layerN++) {
            if (shiftHalfUpLayer[layerN]) {
                out+=outvalset[layerN][i];
            } else {
                out+="       ";
            }
        }
        out+="\n";
        // Print unshifted
        for (let layerN=0; layerN<outvalset.length; layerN++) {
            if (!shiftHalfUpLayer[layerN]) {
                out+=outvalset[layerN][i];
            } else {
                out+="       ";
            }
        }
        out+="\n";
    }
    return out;
}


