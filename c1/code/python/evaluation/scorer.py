from datetime import datetime
import pandas as pd
import numpy as np
import math
import argparse
import os

"""
Crater Detection Data Science Challenge offline scorer
Usage: python scorer.py --pred <path-to-solution> --truth <path-to-truth-file> --out_dir <path-to-output-dir>
See the problem specification for the format of the solution and truth files
Feel free to modify this to suit your needs, and use the contest forum to ask questions.
"""

outDir = ''

XI_2_THRESH = 13.277
NN_PIX_ERR_RATIO = 0.07
    
def calcYmat(a, b, phi):
    unit_1 = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
    unit_2 = np.array([[1 / (a ** 2), 0], [0, 1 / (b ** 2)]])
    unit_3 = np.array([[math.cos(phi), math.sin(phi)], [-math.sin(phi), math.cos(phi)]])
    return unit_1 @ unit_2 @ unit_3

def calc_dGA(Yi, Yj, yi, yj):
    multiplicand = 4 * np.sqrt(np.linalg.det(Yi) * np.linalg.det(Yj)) / np.linalg.det(Yi + Yj)
    exponent = (-0.5 * (yi - yj).T @ Yi @ np.linalg.inv(Yi + Yj) @ Yj @ (yi - yj))
    e = exponent[0, 0]
    cos = multiplicand * np.exp(e)
    cos = min(1, cos)
    return np.arccos(cos)

def dGA(crater_A, crater_B):
    
    A_a = crater_A['ellipseSemimajor(px)']
    A_b = crater_A['ellipseSemiminor(px)']
    A_xc = crater_A['ellipseCenterX(px)']
    A_yc = crater_A['ellipseCenterY(px)']
    A_phi = crater_A['ellipseRotation(deg)'] / 180 * math.pi

    B_a = crater_B['ellipseSemimajor(px)']
    B_b = crater_B['ellipseSemiminor(px)']
    B_xc = crater_B['ellipseCenterX(px)']
    B_yc = crater_B['ellipseCenterY(px)']
    B_phi = crater_B['ellipseRotation(deg)'] / 180 * math.pi

    A_Y = calcYmat(A_a, A_b, A_phi)
    B_Y = calcYmat(B_a, B_b, B_phi)
    
    A_y = np.array([[A_xc], [A_yc]])
    B_y = np.array([[B_xc], [B_yc]])

    dGA = calc_dGA(A_Y, B_Y, A_y, B_y)

    ab_min = np.min([A_a, A_b])
    comparison_sig = NN_PIX_ERR_RATIO * ab_min
    ref_sig = 0.85 / np.sqrt(A_a * A_b) * comparison_sig
    xi_2 = dGA * dGA / (ref_sig * ref_sig)
    
    return dGA, xi_2


def writeScore(s):
    path = os.path.join(outDir, 'result.txt')
    out = open(path, 'w')
    out.write(str(s))
    out.close()

def score1(ts, ps):
    if len(ps) == 0:
        return 0.0
    t_empty = False
    p_empty = False
    if len(ts) == 1 and ts[0].get('ellipseSemimajor(px)') == -1:
        t_empty = True
    if len(ps) == 1 and ps[0].get('ellipseSemimajor(px)') == -1:
        p_empty = True
    if t_empty and p_empty:
        return 1.0
    if t_empty != p_empty:
        return 0.0
    
    dgas = []

    for t in ts:
        # find best matching prediction
        best_p = None
        best_dGA = math.pi / 2
        best_xi_2 = float('inf')

        for p in ps:
            if p['matched']:
                continue
            # short-circuit checks
            rA = min(t['ellipseSemimajor(px)'], t['ellipseSemiminor(px)'])
            rB = min(p['ellipseSemimajor(px)'], p['ellipseSemiminor(px)'])
            if rA > 1.5 * rB or rB > 1.5 * rA:
                continue
            r = min(rA, rB )
            if abs(t['ellipseCenterX(px)'] - p['ellipseCenterX(px)']) > r:
                continue
            if abs(t['ellipseCenterY(px)'] - p['ellipseCenterY(px)']) > r:
                continue
            d, xi_2 = dGA(t, p)
            if d < best_dGA:
                best_dGA = d
                best_p = p
                best_xi_2 = xi_2
        if best_xi_2 < XI_2_THRESH: # matched
            t['matched'] = True
            best_p['matched'] = True
            dgas.append(1 - best_dGA / math.pi)

    if len(dgas) == 0:
        return 0.0        
    avg_dga = sum(dgas) / len(ps)
    tp_count = len(dgas)
    ret = avg_dga * min(1.0, tp_count / min(10, len(ts)))
    return ret

def main():
    parser = argparse.ArgumentParser()
    # ignore the defaults here, or rewrite them to match your setup
    parser.add_argument('--truth', type=str, default='./gt-files/test-gt.csv', help='path to gt file')
    parser.add_argument('--pred', type=str, default='./sample-data/solution.csv', help='path to prediction file')
    parser.add_argument('--out_dir', type=str, default='./scorer-out/', help='output folder')
    args = parser.parse_args()

    global outDir
    outDir = args.out_dir
    os.makedirs(outDir, exist_ok=True)

    try:
        print('Reading truth data from', args.truth)
        truth_df = pd.read_csv(args.truth)
    except Exception as e:
        print('Error reading truth: ', str(e))
        writeScore(-1)
        exit(-10)
    try:
        print('Reading detections from', args.pred)
        detections_df = pd.read_csv(args.pred)
    except Exception as e:
        print('Error reading detections: ', str(e))
        writeScore(-1)
        exit(-11)

    truth_df['matched'] = False
    detections_df['matched'] = False
    truth = (
    truth_df.set_index('inputImage').groupby(level='inputImage')
      .apply(lambda g: g.to_dict(orient='records'))
      .to_dict()
    )
    detections = (
    detections_df.set_index('inputImage').groupby(level='inputImage')
      .apply(lambda g: g.to_dict(orient='records'))
      .to_dict()
    )

    image_ids = list(truth.keys())
    sum = 0
    for id in image_ids:
        truth_craters = truth[id]
        if id not in detections:
            continue
        detection_craters = detections[id]
        try:
            score = score1(truth_craters, detection_craters)
            # print(f'Image {id}: score {score}') # TODO remove
        except Exception as e:
            print(f'Error in detections with image {id}: {str(e)}')
            writeScore(-1)
            exit(-2)
        sum += score

    sum = sum / len(image_ids)
    sum = 100 * sum
    print(f'Score: {sum}')
    writeScore(sum)

if __name__ == '__main__':
    main()