**What the Test Is
**
The experiment is a browser-based cognitive task that measures how people make decisions under different levels of time pressure. Participants complete 40 trials across four conditions — combining low and high time pressure with low and high task complexity. Each trial presents either a cognitive interference task (such as a Stroop color-word conflict, a flanker attention task, or a numerical size-value conflict) or a risk-based decision scenario (such as choosing between a guaranteed reward and a probabilistic gamble).

**Why These Tasks
**
These specific task types were chosen because they create a measurable conflict between System 1 (fast, automatic, intuitive thinking) and System 2 (slow, deliberate, analytical thinking). Under low time pressure, participants can override their automatic response and think carefully. Under high time pressure, the automatic response takes over. The moment that switch happens — from analytical to intuitive — is what the experiment is designed to capture through response times and choice patterns.

**Data Collection
**
The answers get collected in this Google Sheets: https://docs.google.com/spreadsheets/d/1joKfkOvM1FKgw7fbe8N2aRzGjUwB1q10c8OCrq29Gd8/edit?gid=854162145#gid=854162145

**What Happens After Data Collection
**
Once sufficient responses are collected, the data will be cleaned and processed in Python, behavioral features will be engineered per trial, a Hidden Markov Model will be fitted per participant to detect strategy shift points, and a mixed-effects regression model will be built to predict those shifts from environmental variables. 
