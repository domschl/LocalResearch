"use strict";
// Indralib shared ES6 module
import { IndraTime } from './indra_time.js';


// Generate UUID v4
export function uuidv4() {
    const crypto = window.crypto || window.msCrypto;

    if (crypto && crypto.getRandomValues) {
        const buf = new Uint16Array(8);
        crypto.getRandomValues(buf);
        buf[3] = (buf[3] & 0xfff) | 0x4000;
        buf[4] = (buf[4] & 0x3fff) | 0x8000;
        return `${pad4(buf[0])}${pad4(buf[1])}-${pad4(buf[2])}-${pad4(buf[3])}-${pad4(buf[4])}-${pad4(buf[5])}${pad4(buf[6])}${pad4(buf[7])}`;
    } else {
        console.error('Your browser does not support a secure random number generator.');
        return null;
    }
}

function pad4(num) {
    let ret = num.toString(16);
    while (ret.length < 4) {
        ret = `0${ret}`;
    }
    return ret;
}

export class IndraEvent {
    constructor() {
        this.domain = "";
        this.from_id = "";
        this.uuid4 = uuidv4();
        this.parent_uuid4 = "";
        this.seq_no = 0;
        this.to_scope = "";
        this.time_jd_start = IndraTime.datetimeToJulian(new Date());
        this.data_type = "";
        this.data = "";
        this.auth_hash = "";
        this.time_jd_end = null;
    }

    version() {
        return "02";
    }

    old_versions() {
        return ["", "01"];
    }

    to_dict() {
        return { ...this };
    }

    to_json() {
        return JSON.stringify(this);
    }

    static from_json(json_str) {
        const ie = new IndraEvent();
        Object.assign(ie, JSON.parse(json_str));
        return ie;
    }

    static mqcmp(pub, sub) {
        for (const c of ["+", "#"]) {
            if (pub.includes(c)) {
                console.log(`Illegal char '${c}' in pub in mqcmp!`);
                return false;
            }
        }
        let inds = 0;
        let wcs = false;
        for (let indp = 0; indp < pub.length; indp++) {
            if (wcs === true) {
                if (pub[indp] === "/") {
                    inds++;
                    wcs = false;
                }
                continue;
            }
            if (inds >= sub.length) {
                return false;
            }
            if (pub[indp] === sub[inds]) {
                inds++;
                continue;
            }
            if (sub[inds] === "#") {
                return true;
            }
            if (sub[inds] === "+") {
                wcs = true;
                inds++;
                continue;
            }
            if (pub[indp] !== sub[inds]) {
                return false;
            }
        }
        if (sub.slice(inds).length === 0) {
            return true;
        }
        if (sub.slice(inds).length === 1) {
            if (sub[inds] === "+" || sub[inds] === "#") {
                return true;
            }
        }
        return false;
    }
}

