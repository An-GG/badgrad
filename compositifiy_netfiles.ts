import fs from 'fs';

type Netfile = {
    iterations: {[n:string]:any }
}
let netfile : Netfile = { iterations: {} };

let fnames = fs.readdirSync("temp_netfiles");
for (let fn of fnames) {
    let f = JSON.parse(fs.readFileSync('temp_netfiles/'+fn.substring(0, fn.length - '.json'.length)).toString());
    netfile.iterations[fn] = f;
}

fs.writeFileSync('viewer/netfile.json', JSON.stringify(netfile));
