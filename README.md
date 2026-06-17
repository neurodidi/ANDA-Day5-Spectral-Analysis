# ANDA Day5 Spectral Analysis

## Staff

 - **Lead Trainer**:
   - Atle E. Rimehaug, Uni Bonn, Germany
 - **Lecturers**: 
   - Udo Ernst, Universität Bremen, Germany
 - **Teaching Assistants**: 
    - Ole Bialas, Uni Bonn, Germany
    - Cristiano Köhler, Forchungszentrum Jülich, Germany
    - Anton Chizhov, Universität Bremen, Germany
    - Michael Denker, Forchungszentrum Jülich, Germany
    - Nicholas Del Grosso, Uni Bonn, Germany
 

## Session Overview

How can we tell which rhythms are present in a neural recording, when they
occur, and whether two brain regions are oscillating together?

In this hands-on session, we will explore methods for analysing neural signals
in the frequency domain. We will work with toy signals that we build ourselves,
where we know exactly which frequencies went in, as well as synthetic local
field potential recordings from visual cortex.

We will begin by constructing periodic signals and using the Fourier transform
to recover their frequency content, then move to the power spectral density and
see how sampling rate, recording duration and averaging shape the spectra we
estimate. Because the Fourier transform assumes that frequency content does not
change over time, it cannot tell us *when* a rhythm appears, so we will turn to
the Morlet wavelet transform for time-resolved spectra and discuss which parts
of the result are trustworthy and which are contaminated by edge artefacts.

Later, we will move beyond single signals and ask whether two regions are
coupled. We will use spectral coherence to find shared rhythms and their time
lags, and the Hilbert transform and phase locking value to measure how
consistently two signals hold a fixed phase relationship.

The rest of the day's materials, including exercise, notebooks, datasets, and solutions will appear in this repo during the session.  Just call `git pull` and you'll get the updates.

See you there!
