# BBO_Capstone_Project
Bayesian Black-Box Optimization Capstone Project


BBO (Black-Box Optimisation) Capstone Project
Section 1: Project Overview

The BBO capstone project focuses on the optimisation of unknown functions where only inputs and outputs are available. Each function is treated as a black box: its analytical form, gradients, and noise characteristics are unknown. Information about the function can be gained only through sequential query feedback.

Main Goal:
The goal is to maximize the output of multiple unknown functions while operating under strict query constraints. This setup mirrors real-world scenarios in machine learning, such as hyperparameter tuning, simulation-based optimisation, and experimental design, where evaluations are expensive and access to the function is limited. The project demonstrates how iterative modelling and principled optimisation can support decision-making under uncertain conditions.

Career Relevance:
This project develops practical skills in:

Bayesian optimisation

Uncertainty-aware modelling

Iterative decision-making

These skills are directly applicable to data science, ML engineering, and research roles, where optimising complex systems efficiently is essential.

Section 2: Inputs and Outputs

Inputs:

Query points are vectors of continuous values representing candidate solutions.

Each value lies in [0, 1] and is specified to six decimal places.

Dimensionality varies by function (from 2D to 8D).

Input format: hyphen-separated string, for example:

2D: 0.372451-0.684219

3D: 0.123456-0.654321-0.234567

8D: 0.500000-0.250000-0.750000-0.500000-0.250000-0.750000-0.500000-0.250000

Outputs:

A single scalar value representing the function output at the queried point.

The scale, smoothness, and noise of outputs are unknown and function-specific.

Outputs are used solely to inform subsequent modelling and query decisions.

Constraints:

Only one query per function per iteration is allowed.

Initial datasets contain ten observations per function.

Section 3: Challenge Objectives

The primary objective is to maximize each unknown function over successive iterations.

Key Constraints:

Limited total number of queries (e.g., 1 new query per function per week).

Sequential feedback: outputs are available only after submission.

Function structures and noise characteristics are unknown.

Increasing dimensionality leads to sparse sampling and higher uncertainty.

Optimization Challenge:
The project requires carefully balancing exploration and exploitation: learning about the function landscape while refining promising regions, especially in high-dimensional spaces or when the query budget is limited.

Section 4: Technical Approach

Strategy Evolution (Weeks 1–3):

Week 1 – Structured Exploration:
Initial queries focused on broad coverage using space-filling methods (e.g., Latin Hypercube Sampling, LHS). The goal was to explore the input space evenly and reduce bias toward boundaries.

Week 2 – Adaptive Exploration vs Exploitation:
Based on Week 1 results, promising regions were identified. Exploration was maintained in high-dimensional spaces, while low-dimensional promising areas were refined for potential optima.

Week 3 – Model-Driven Bayesian Optimisation:
A Gaussian Process (GP) surrogate was fitted for each function. Kernel selection and hyperparameters were optimised automatically. Query points were selected using acquisition functions (e.g., Upper Confidence Bound, Expected Improvement) with explicit control of the exploration–exploitation trade-off.

Methods Used:

Gaussian Process regression with RBF and Matérn kernels

Automatic kernel and hyperparameter selection via marginal likelihood maximization

Expected Improvement (EI) and UCB acquisition functions

Exploration parameters (ξ for EI, β for UCB) tuned for each function and dimensionality

Exploration vs Exploitation:
Regions of high uncertainty are explored more aggressively, while promising regions are exploited to refine optima. Parameter tuning is function-specific, not uniform.

Future Extensions:

Support Vector Machines (SVMs) could classify regions as high- or low-performing.

SVMs could complement GP-based methods in high-dimensional spaces or where surrogate regression is less reliable.

Multi-surrogate ensembles and safe optimisation strategies could be explored for risk-sensitive functions.

Key Differentiator:
This approach adapts dynamically to each function’s characteristics, balancing exploration and exploitation based on observed performance, dimensionality, and uncertainty.
