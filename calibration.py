import cv2 as cv
import glob
import numpy as np
import os
from scipy import linalg

# CONFIGURAZIONE PARAMETRI
CONFIG = {
    'frame_width': 1920,
    'frame_height': 1080,
    'checkerboard_rows': 8,             # Angoli interni verticali
    'checkerboard_columns': 11,          # Angoli interni orizzontali
    'checkerboard_box_size_scale': 0.00045, # Dimensione reale del quadratino
    'view_resize': 2,                   # Riduzione per visualizzazione a schermo
    'stereo_calibration_frames': 20     # Numero di coppie di frame da salvare
}

# Percorsi dei video pre-acquistati
VIDEO_PATH_LEFT = '/Users/giuliagalvan/Library/CloudStorage/OneDrive-KULeuven/Calibration Left.avi'
VIDEO_PATH_RIGHT = '/Users/giuliagalvan/Library/CloudStorage/OneDrive-KULeuven/Calibration Right.avi'

# Directory base di output
BASE_DIR = '/Users/giuliagalvan/Library/CloudStorage/OneDrive-KULeuven/summer_school'


def DLT(P1, P2, point1, point2):
    A = [point1[1] * P1[2, :] - P1[1, :],
         P1[0, :] - point1[0] * P1[2, :],
         point2[1] * P2[2, :] - P2[1, :],
         P2[0, :] - point2[0] * P2[2, :]]
    A = np.array(A).reshape((4, 4))
    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices=False)
    return Vh[3, 0:3] / Vh[3, 3]


def capture_and_filter_stereo_frames():
    """Consente all'utente di selezionare e validare interattivamente i frame dai video."""
    frames_dir = os.path.join(BASE_DIR, 'frames_pair')
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir, exist_ok=True)

    # Sostituisci le vecchie righe con queste:
    cap_left = cv.VideoCapture(VIDEO_PATH_LEFT, cv.CAP_AVFOUNDATION)
    cap_right = cv.VideoCapture(VIDEO_PATH_RIGHT, cv.CAP_AVFOUNDATION)

    view_resize = CONFIG['view_resize']
    target_frames = CONFIG['stereo_calibration_frames']
    rows, cols = CONFIG['checkerboard_rows'], CONFIG['checkerboard_columns']
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    saved_count = 0
    print("\n=== MODALITÀ CATTURA INTERATTIVA ===")
    print("1. Guarda il video. Premi 'S' quando vuoi catturare un frame.")
    print("2. In pausa: controlla gli angoli. Premi 'S' per CONFERMARE o 'C' per ANNULLARE.\n")

    while True:
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()

        if not ret_l or not ret_r:
            print("Fine dei file video raggiunta.")
            break

        # Ridimensiona solo per la visualizzazione a schermo
        show_l = cv.resize(frame_l, None, fx=1/view_resize, fy=1/view_resize)
        show_r = cv.resize(frame_r, None, fx=1/view_resize, fy=1/view_resize)

        # Testo informativo a schermo
        cv.putText(show_l, f"Salvati: {saved_count}/{target_frames} | Premi 'S' per bloccare", 
                   (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv.imshow("Telecamera Sinistra (Guida)", show_l)
        cv.imshow("Telecamera Destra", show_r)
        
        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC per uscire
            quit()
            
        if key == ord('s'):  # Richiesta di cattura frame corrente
            print(f"\n[Verifica] Analisi accoppiamento frame potenziale #{saved_count}...")
            
            # Copie di lavoro per non sporcare i frame originali ad alta risoluzione
            check_l = frame_l.copy()
            check_r = frame_r.copy()
            
            gray_l = cv.cvtColor(check_l, cv.COLOR_BGR2GRAY)
            gray_r = cv.cvtColor(check_r, cv.COLOR_BGR2GRAY)
            
            ret_cb_l, corners_l = cv.findChessboardCorners(gray_l, (rows, cols), None)
            ret_cb_r, corners_r = cv.findChessboardCorners(gray_r, (rows, cols), None)

            # Disegna gli angoli per la decisione dell'utente
            if ret_cb_l:
                corners_l = cv.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
                cv.drawChessboardCorners(check_l, (rows, cols), corners_l, ret_cb_l)
            if ret_cb_r:
                corners_r = cv.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
                cv.drawChessboardCorners(check_r, (rows, cols), corners_r, ret_cb_r)

            # Mostra il fermo immagine con l'anteprima degli angoli rilevati
            preview_l = cv.resize(check_l, None, fx=1/view_resize, fy=1/view_resize)
            preview_r = cv.resize(check_r, None, fx=1/view_resize, fy=1/view_resize)
            
            cv.putText(preview_l, "CONFERMI? 'S' = Si / 'C' = No", (20, 40), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv.imshow("Telecamera Sinistra (Guida)", preview_l)
            cv.imshow("Telecamera Destra", preview_r)

            # Ciclo di attesa decisione dell'utente
            while True:
                choice = cv.waitKey(0) & 0xFF
                if choice == ord('s'):  # Conferma e Salva
                    if ret_cb_l and ret_cb_r:
                        cv.imwrite(os.path.join(frames_dir, f'frames_pair_left_{saved_count}.png'), frame_l)
                        cv.imwrite(os.path.join(frames_dir, f'frames_pair_right_{saved_count}.png'), frame_r)
                        print(f"-> Coppia {saved_count} SALVATA con successo.")
                        saved_count += 1
                    else:
                        print("-> SCARTATO AUTOMATICAMENTE: Scacchiera non rilevata in una delle due camere.")
                    break
                elif choice == ord('c'):  # Scarta e prosegui
                    print("-> Scartato dall'utente.")
                    break
                elif choice == 27:
                    quit()

        if saved_count == target_frames:
            print("\nTarget di frame di calibrazione raggiunto.")
            break

    cap_left.release()
    cap_right.release()
    cv.destroyAllWindows()


def calibrate_camera_for_intrinsic_parameters(images_prefix):
    """Calcola i parametri intrinseci basandosi sui frame validati e salvati."""
    images_names = sorted(glob.glob(images_prefix))
    if not images_names:
        print(f"ERRORE: Nessun frame trovato in: {images_prefix}")
        quit()

    images = [cv.imread(imname, 1) for imname in images_names]
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    rows, cols = CONFIG['checkerboard_rows'], CONFIG['checkerboard_columns']
    world_scaling = CONFIG['checkerboard_box_size_scale']

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
    objp = world_scaling * objp

    width, height = images[0].shape[1], images[0].shape[0]
    imgpoints, objpoints = [], []

    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (rows, cols), None)
        if ret:
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, cmtx, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    return cmtx, dist


def save_camera_intrinsics(camera_matrix, distortion_coefs, right_flag):
    param_dir = os.path.join(BASE_DIR, 'camera_parameters')
    os.makedirs(param_dir, exist_ok=True)
    side = "right" if right_flag == 1 else "left"
    out_filename = os.path.join(param_dir, f'camera_parameters_intrinsics_{side}.dat')

    with open(out_filename, 'w') as outf:
        outf.write('intrinsic:\n')
        for row in camera_matrix:
            outf.write(" ".join(map(str, row)) + "\n")
        outf.write('distortion:\n')
        outf.write(" ".join(map(str, distortion_coefs[0])) + "\n")


def stereo_calibrate(mtx_left, dist_left, mtx_right, dist_right, frames_prefix_left_pair, frames_prefix_right_pair):
    c0_images_names = sorted(glob.glob(frames_prefix_left_pair))
    c1_images_names = sorted(glob.glob(frames_prefix_right_pair))

    c0_images = [cv.imread(imname, 1) for imname in c0_images_names]
    c1_images = [cv.imread(imname, 1) for imname in c1_images_names]

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 1000, 0.0001)
    rows, cols = CONFIG['checkerboard_rows'], CONFIG['checkerboard_columns']
    world_scaling = CONFIG['checkerboard_box_size_scale']

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
    objp = world_scaling * objp

    width, height = c0_images[0].shape[1], c0_images[0].shape[0]
    imgpoints_left, imgpoints_right, objpoints = [], [], []

    for frame0, frame1 in zip(c0_images, c1_images):
        gray1 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, cols), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, cols), None)

        if c_ret1 and c_ret2:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, _, _, _, _, R, T, _, _ = cv.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right, mtx_left, dist_left,
        mtx_right, dist_right, (width, height), criteria=criteria, flags=stereocalibration_flags
    )
    print('Stereo RMSE finale: ', ret)
    return R, T


def save_extrinsic_calibration_parameters(R0, T0, R1, T1):
    param_dir = os.path.join(BASE_DIR, 'camera_parameters')
    os.makedirs(param_dir, exist_ok=True)
    for idx, (R_mat, T_vec) in enumerate([(R0, T0), (R1, T1)]):
        filename = os.path.join(param_dir, f'camera{idx}_rot_trans.dat')
        with open(filename, 'w') as outf:
            outf.write('R:\n')
            for row in R_mat:
                outf.write(" ".join(map(str, row)) + "\n")
            outf.write('T:\n')
            for row in T_vec:
                outf.write(" ".join(map(str, row)) + "\n")


if __name__ == '__main__':
    # FASE 1: Selezione e filtraggio interattivo dei frame dai video ad opera dell'utente
    capture_and_filter_stereo_frames()

    print("\n--- FASE 2: Calibrazione Intrinseci dai frame salvati ---")
    # Generazione dei percorsi partendo dai frame appena salvati nella cartella condivisa 'frames_pair'
    prefix_left = os.path.join(BASE_DIR, 'frames_pair', 'frames_pair_left_*')
    cmtx0, dist0 = calibrate_camera_for_intrinsic_parameters(prefix_left)
    save_camera_intrinsics(cmtx0, dist0, right_flag=0)
    
    prefix_right = os.path.join(BASE_DIR, 'frames_pair', 'frames_pair_right_*')
    cmtx1, dist1 = calibrate_camera_for_intrinsic_parameters(prefix_right)
    save_camera_intrinsics(cmtx1, dist1, right_flag=1)

    print("\n--- FASE 3: Calibrazione Stereo Extrinsics ---")
    R, T = stereo_calibrate(cmtx0, dist0, cmtx1, dist1, prefix_left, prefix_right)

    print("\n--- FASE 4: Salvataggio Matrici di Sistema Estrinseche ---")
    R0 = np.eye(3, dtype=np.float32)
    T0 = np.array([0., 0., 0.]).reshape((3, 1))
    save_extrinsic_calibration_parameters(R0, T0, R, T)
    
    print("\nProcesso di calibrazione interattiva completato con successo!")