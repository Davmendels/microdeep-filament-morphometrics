# Microdeep Filament Morphometrics

## 1 Introduction

Filamentous microorganisms such as *Arthrospira* form elongated trichomes that are difficult to characterize using conventional particle analysis methods.

This note describes a simple morphometric estimator based on the relationship between the contour area and perimeter of isolated filaments.

---

## 2 Imaging assumptions

Filamentous microorganisms such as *Arthrospira* form elongated trichomes with approximately constant diameter. When imaged using brightfield microscopy, individual filaments can be segmented as contiguous objects in the image plane.

For the morphometric estimator described here to be valid the following conditions are assumed:

- the filament is **isolated** and does not overlap with other filaments
- the filament diameter is **approximately constant** along its length
- the segmentation correctly captures the filament contour

Under these conditions, the projected filament shape can be described by two geometric observables extracted from the segmented contour:

- the **area** $A$
- the **perimeter** $P$

These quantities can be obtained directly from the binary segmentation mask.

---

## 3 Segmentation

Filament detection proceeds in three main steps.

1. **Object detection**

   A convolutional neural network (YOLO) is used to detect candidate filament regions in the image. Each detected region defines a region of interest (ROI).

\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{Figure1.png}
\caption{YOLO training and inference workflow used to detect filament objects.}
\end{figure}

2. **Adaptive thresholding**

   Within each ROI, adaptive thresholding is applied to obtain a binary segmentation of the filament.

   Following adaptive thresholding, a simple morphological closing operation (dilation followed by erosion) is applied to stabilize the binary mask and reduce small discontinuities caused by poor local contrast.

   Candidate contours are then extracted from the mask and ranked by area. Because the filament of interest is expected to dominate the ROI provided by the detector, only the largest contour is retained for subsequent morphometric analysis.

   Additional rejection criteria, such as minimum area and minimum aspect ratio, can be applied to exclude spurious contours.

3. **Contour extraction**

   The largest connected component in the binary mask is retained and its contour is extracted. From this contour, the following geometric descriptors are computed:

   - area $A$
   - perimeter $P$

These two quantities are sufficient to estimate filament length and diameter using the geometric estimator described below.

\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{Figure2.png}
\caption{Detection and morphometric extraction of Spirulina filaments using YOLO bounding boxes and adaptive threshold segmentation.}
\end{figure}

---

## 4 Geometric estimator

An isolated filament can be approximated as a slender elongated object of length $L$ and diameter $D$.

To first order, the projected area of such an object can be written as

$$
A \approx L D
$$

while the contour perimeter is approximated by

$$
P \approx 2(L + D)
$$

These expressions correspond to the geometric properties of a rectangular approximation of the filament contour.

Substituting

$$
D = \frac{A}{L}
$$

into the perimeter expression gives

$$
P = 2\left(L + \frac{A}{L}\right)
$$

Multiplying both sides by $L$ yields

$$
PL = 2L^2 + 2A
$$

which can be rearranged into the quadratic equation

$$
L^2 - \frac{P}{2}L + A = 0
$$

Solving this equation gives two roots

$$
L, D = \frac{P}{4} \pm \frac{1}{4}\sqrt{P^2 - 16A}
$$

The larger root corresponds to the filament **length** $L$, while the smaller root corresponds to the effective filament **diameter** $D$.

Because the estimator depends only on area and perimeter, it remains computationally simple and robust even when the filament exhibits moderate curvature.

---

## 5 Limitations

The geometric estimator described above relies on the assumption that the segmented object corresponds to a **single isolated filament** of approximately constant diameter. When these conditions are satisfied, the
perimeter–area relationship provides a robust estimate of filament length and diameter.

In practice, three regimes can be distinguished.

### 5.1 Isolated filaments

In the ideal case, the filament is completely isolated and does not touch any other object in the field of view. The segmentation then produces a single elongated contour whose area $A$ and perimeter $P$ correspond to the geometric assumptions of the model.

In this regime the estimator provides reliable measurements of both filament length and effective diameter.

### 5.2 Touching filaments

Filaments may occasionally touch or come into close proximity without forming significant overlaps. In such cases the segmentation may still produce a single connected contour, but the geometric assumptions of the model are only approximately satisfied.

Measurements obtained in this regime may exhibit moderate bias and should therefore be interpreted with caution. Simple geometric filters, such as aspect ratio thresholds or contour compactness metrics, can be used to exclude ambiguous cases.

### 5.3 Overlapping filaments

When filaments cross or overlap in the image plane, the segmentation mask no longer corresponds to a single elongated object. The resulting contour combines multiple filaments and therefore violates the geometric assumptions of the estimator.

In this situation the perimeter–area relationship cannot uniquely resolve filament length and diameter. Such objects must therefore be rejected from the morphometric analysis.

### 5.4 Sample dilution

Reliable morphometric measurements are obtained when the sample is sufficiently dilute such that individual filaments can be segmented unambiguously. In practice, adjusting the dilution of the sample prior to imaging is often the most effective way to ensure that the estimator operates within its validity domain.

---

## 6 Output units

The reference implementation reports filament measurements in image units,
namely pixels for length and diameter, and pixels squared for area.

This choice is deliberate. The physical conversion from pixels to metric
units depends on the full imaging configuration, including objective
magnification, camera resolution, sensor mode, and any resizing applied
during image acquisition or preprocessing.

Users who know the calibration factor $s$ expressed in $\um/\px$ can
convert the measurements afterward according to

$$
L_{\um} = s\,L_{\px}
$$

and

$$
D_{\um} = s\,D_{\px}
$$

The present implementation therefore separates geometric measurement from
instrument-specific calibration.

---

## 7 Example workflow/results

## 7.1 Example workflow

The complete morphometric pipeline therefore consists of the following steps:

1. Detection of candidate filaments using a YOLO network.
2. Extraction of a region of interest (ROI) for each detection.
3. Adaptive thresholding within the ROI.
4. Contour extraction from the binary mask.
5. Computation of area $A$ and perimeter $P$.
6. Estimation of filament length $L$ and diameter $D$ using the geometric estimator.

This hybrid approach combines the robustness of modern object detection methods
with a simple geometric model to produce reliable filament measurements.

## 7.2 Example results

The pipeline was tested on representative microscopy images of Arthrospira filaments acquired using the Microdeep system.

The YOLO detector successfully identifies filament ROIs, which are then processed using adaptive thresholding and morphological closing. Contours are extracted and the largest valid contour within each ROI is used to compute area and perimeter.

Filament length and diameter are estimated using the geometric model described in Section 4.

The complete processing pipeline can be executed using:

```
python code/detect_and_measure.py \
  --input data/images \
  --weights data/model/filaments.pt \
  --output results.csv \
  --save-rois roi_out \
  --save-masks mask_out
  ```