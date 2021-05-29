
let K_LAYERS = [2, 5, 5, 2];

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
            value:0 // inital value of nodes doesn't matter
        }

        if (layerN > 0) {
            // Add links to prev layer
            for (let prev_node of net[layerN-1]) {
                let newlink:link = {
                    weight: 0,
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

