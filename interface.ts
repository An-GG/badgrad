import fs from 'fs';
import { promisify } from 'util';


export class MnistReader<T extends "LABELS" | "IMAGES"> {

    public setType: T;
    public metadata: T extends "LABELS" ? 
        {magicN:number, length:number} : 
        {magicN:number, length:number, res: {x:number, y:number, n_pixels:number} } = {} as any;
    private s: fs.ReadStream;
    public currentIndex = 0;

    private fpaths = {
        "REL":"mnist/",
        "TRAINING": {
            "IMAGES":"train-images.idx3-ubyte",
            "LABELS":"train-labels-idx1-ubyte"
        },
        "TEST": {
            "IMAGES":"t10k-images.idx3-ubyte",
            "LABELS":"t10k-labels.idx1-ubyte"
        }
    } as const;

    private constructor(set: "TRAINING" | "TEST", type: T) {
        this.setType = type;
        this.s = fs.createReadStream(this.fpaths.REL + this.fpaths[set][type]);
    }

    public static async getReader<SetType extends "LABELS" | "IMAGES">(set: "TRAINING" | "TEST", type: SetType):Promise<MnistReader<SetType>> {
        let r = new MnistReader<SetType>(set, type);
        await r.setup();
        (r as MnistReader<"LABELS">).metadata = {
            magicN: (r.s.read(4) as Buffer).readInt32BE(),
            length: (r.s.read(4) as Buffer).readInt32BE(),
        }
        if (type == "IMAGES") {
            (r as MnistReader<"IMAGES">).metadata.res = {
                y: (r.s.read(4) as Buffer).readInt32BE(),
                x: (r.s.read(4) as Buffer).readInt32BE(),
                n_pixels:0
            };
            
            (r as MnistReader<"IMAGES">).metadata.res.n_pixels = 
                (r as MnistReader<"IMAGES">).metadata.res.x * (r as MnistReader<"IMAGES">).metadata.res.y;
        }
        return r;
    }

    private async setup() {
        let asyncStreamOnce = promisify(
            (event: string, cb:(err?:Error)=>void)=>{
                this.s.once(event, ()=>{ cb(undefined); })
            }
        );
        await asyncStreamOnce("readable");
    }

    public next():  T extends "LABELS" ? number : number[] {
        this.currentIndex++;
        if (this.setType == "LABELS") {
            return (this.s.read(1) as Buffer).readUInt8() as any;
        } else {
            let v:number[] = [];
            let n_px = (this as MnistReader<"IMAGES">).metadata.res.n_pixels;
            for (let i = 0; i < n_px; i++) {
                this.currentIndex++;
                v.push((this.s.read(1) as Buffer).readUInt8());
            }
            return v as any;
        }
    }

    public shiftHead(n:number) {

    }

    public vectorToASCII(v: number[]):string {
        const ASCIIScale = " .:-=+*#%@".split('');
        let out = "";
        for (let i=0; i<v.length; i++) { 
            // map 10 to 8bit 256
            let p = v[i];
            out+=(ASCIIScale[Math.floor(p * (10/256))]);
            if ((i+1)%(this as MnistReader<"IMAGES">).metadata.res.x == 0) {
                out+="\n";
            }
        }
        return out;
    } 
}


