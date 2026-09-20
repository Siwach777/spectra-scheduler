# Project Scope: Passive Smart Scan Strategy for Wideband Spectrum Reception

> [!IMPORTANT]
> **Scope Boundary: Passive / Receive-Only**
> This project develops software for scheduling observations by a passive receiver across a wide frequency range. The system only models reception, observation, prediction, and scheduling. It does not require transmitting signals or controlling emitters.

## 1. Problem Background

Detection of signals of interest requires searching a frequency spectrum wider than the instantaneous bandwidth of the receiver. Because the receiver cannot observe the entire frequency range at the same time, it must scan or sweep across multiple frequency bands.

Traditional open-loop scan strategies rely on pre-mission or prior information and generally prioritize rapid coverage of the complete frequency range. When prior information is unavailable or unreliable, a fixed scan strategy may allocate observation time inefficiently and may fail to observe newly active or changing emitters at the correct frequency and time.

The project therefore treats interception as a two-dimensional frequency-time search problem: the receiver must observe the appropriate frequency band at the appropriate time.

## 2. Project Objective

The objective is to develop a machine-learning-based passive receiver scheduler that operates without reliable prior knowledge of emitter operating characteristics.

The scheduler must use observations obtained during scanning to adapt future scan decisions, with the aim of:

- minimizing intercept time;
- maintaining a high interception rate;
- adapting its scan behavior using observed hits and misses; and
- continuing to search across the available frequency bands without relying on fixed prior emitter information.

## 3. Required System Model

A receiver system model shall be developed for evaluating the scan strategy.

The modeled receiver operates across a frequency spectrum divided into multiple bands. At each time step, the receiver observes only the band or bands permitted by its instantaneous bandwidth.

The scan problem therefore includes decisions about:

- which frequency band to observe;
- when that band should be observed; and
- how observation time should be allocated across the available bands.

The receiver model is used only to support the passive scheduling and evaluation problem described in the problem statement.

## 4. Simulated RF Environment

The scheduler shall be developed and evaluated using a simulated RF environment containing truth information about emitter activity.

For each frequency band and each time step, the environment records whether a transmission is present or absent:

$$S(k,t) \in \{0,1\}$$

where $k$ identifies a frequency band and $t$ identifies a time step.

This truth information is used to train and evaluate the model against the known simulated environment. The scheduling process itself is based on the observations available to the receiver.

## 5. Emitter Behaviors Required by the Problem Statement

The system model shall support evaluation against the emitter behaviors explicitly identified in the problem statement:

- **Frequency-agile emitters**, whose operating frequency can change over time.
- **Spatially scanning emitters**, whose observable transmission opportunities vary with time.
- **Periodic scan emitters**, for which approaches to improve interception must be developed and evaluated.

The scheduler must operate without assuming reliable prior knowledge of these emitters' operating characteristics.

## 6. Machine-Learning-Based Scheduler

The primary software component is a robust machine-learning-based scheduler for the passive receiver.

The scheduler shall:

- use receiver observations from previous scan actions;
- learn from hits and misses;
- determine future scan scheduling decisions across frequency and time;
- support prediction of intercept time;
- support evaluation of interception ratio; and
- adapt its scheduling behavior as observations accumulate.

The problem statement does not prescribe a specific machine-learning algorithm. The selected algorithm or algorithms are implementation choices to be evaluated against the required figures of merit.

## 7. Required Figures of Merit

Performance shall be evaluated using the figures of merit explicitly listed in the problem statement:

- Probability of Detection
- Probability of False Alarm
- Sensitivity
- Average Intercept Rate
- Average Reward / Cost Function
- Percentage of Correct Predictions
- Average Intercept Time Error

The problem statement does not define specific mathematical formulations for these metrics. Their exact computation shall therefore be documented as part of the implementation rather than treated as a prescribed requirement.

## 8. Intercept-Time and Interception-Ratio Prediction

The receiver model and scheduler shall enable prediction and evaluation of:

- intercept time; and
- interception ratio

against spatially scanning and frequency-agile emitters.

These predictions shall be compared with the truth information available in the simulated environment for performance evaluation.

## 9. Periodic Scan Interception

The project shall outline and develop approaches for improving interception of periodic scan emitters.

The problem statement does not prescribe a particular technique for this requirement. Candidate algorithms shall therefore be treated as implementation choices and evaluated using the required performance measures.

## 10. Expected Solution

The expected solution is a machine-learning-based passive receiver scheduler software system consisting of:

- a passive receiver scheduling model;
- a simulated RF environment with per-band, per-time transmission truth information;
- an adaptive scheduler trained from observed hits and misses;
- prediction of intercept time and interception ratio;
- support for spatially scanning, frequency-agile, and periodic scan emitter scenarios; and
- evaluation using the required figures of merit.

## 11. Requirements Traceability

| Problem Statement Requirement | Scope Coverage |
| :--- | :--- |
| Absence of reliable prior emitter intelligence | Scheduler operates without assuming reliable prior operating characteristics. |
| Receiver instantaneous bandwidth smaller than total spectrum | Receiver scans across multiple frequency bands over time. |
| Frequency-time interception problem | Scheduling is modeled across both frequency and time. |
| Receiver system model | Passive receiver scheduling model is included. |
| Simulated RF environment | Simulated environment is included. |
| Truth information for each band and time step | Transmission / non-transmission state is recorded for each band and time step. |
| Prediction of intercept time | Included as a required scheduler/model capability. |
| Prediction/evaluation of interception ratio | Included as a required scheduler/model capability. |
| Spatially scanning emitters | Included as a required evaluation scenario. |
| Frequency-agile emitters | Included as a required evaluation scenario. |
| Machine-learning scheduler | Core expected software component. |
| Minimize intercept time | Included as a scheduler objective. |
| High interception rate | Included as a scheduler objective. |
| Training from hits and misses | Included as required learning feedback. |
| Probability of detection | Included as a required figure of merit. |
| Probability of false alarm | Included as a required figure of merit. |
| Sensitivity | Included as a required figure of merit. |
| Average intercept rate | Included as a required figure of merit. |
| Average reward / cost function | Included as a required figure of merit. |
| Percentage of correct predictions | Included as a required figure of merit. |
| Average intercept time error | Included as a required figure of merit. |
| Periodic scan interception | Approaches and algorithms must be developed and evaluated. |
| Expected solution | ML-based passive receiver scheduler software. |

## Scope Summary

The project is limited to passive observation, simulation, prediction, learning, scheduling, and evaluation. The technical implementation remains intentionally open wherever the problem statement does not prescribe a specific model, metric formula, or machine-learning algorithm.
