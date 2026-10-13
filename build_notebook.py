"""Generate the intro EEG notebook. Usage: python build_notebook.py OUT.ipynb"""
import sys
import nbformat as nbf

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text, form=False):
    c = nbf.v4.new_code_cell(text.strip("\n"))
    if form:
        c.metadata["cellView"] = "form"
    cells.append(c)


def answer(text):
    """Markdown cell with a collapsible answer."""
    md(
        "<details><summary><b>Answer (open after you've thought about it)</b></summary>\n\n"
        + text.strip("\n")
        + "\n\n</details>"
    )


# ----------------------------------------------------------------------------
md(r"""
# Intro to EEG: what are we actually looking at?
**UChicago Neurotech · Meeting 1**

Today we'll take a real recording of someone's brain activity and clean it up step by step.
By the end you should be able to explain, in plain English, **what each step of a basic EEG pipeline does and why we do it.**

**You don't need to know Python.** All the code is written for you. Read it like a recipe and work out what it's doing. The `# comments` on the right are hints.

### How this works
1. Run a cell with **Shift + Enter** (or the play button on its left).
2. When you see **Predict**, *stop*. Don't run the next cell yet. Think about it and talk it over with your table.
3. Answers are hidden under **Answer**. Open it once you've made your guess.
4. Cells titled **Setup** hide code you don't need to read. Just run them.

*Lost or something broke?* Go to **Runtime → Run all** and you'll be caught up.
""")

code(r'''
#@title Setup: install tools and download data (just run this, takes ~1 minute)
%pip install -q mne==1.13.2

import numpy as np
import matplotlib.pyplot as plt
import mne

mne.set_log_level("ERROR")
mne.viz.set_browser_backend("matplotlib")
plt.rcParams.update({"figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False})


def load_eegbci(subject, run):
    """Load one recording from the PhysioNet EEG Motor Movement/Imagery dataset."""
    [path] = mne.datasets.eegbci.load_data(subject, [run], update_path=True)
    raw = mne.io.read_raw_edf(path, preload=True)
    mne.datasets.eegbci.standardize(raw)    # tidy channel names, e.g. "Fp1." -> "Fp1"
    raw.set_montage("spherical_1005")        # where each electrode sits on the head
    return raw


def load_erp_core():
    """Load 60 s of the ERP CORE recording, which has electrodes around the eyes."""
    path = mne.datasets.erp_core.data_path(update_path=True)
    raw = mne.io.read_raw_fif(path / "ERP-CORE_Subject-001_Task-Flankers_eeg.fif")
    return raw.crop(0, 60).load_data()


def _good_eeg(raw, channels):
    return raw.copy().pick(channels if channels is not None else "eeg", exclude="bads")


def show_traces(raw, channels, start=0, duration=10, title=None, spacing=None, ax=None):
    """Plot a few channels stacked on top of each other, in microvolts."""
    data, times = raw.get_data(picks=channels, tmin=start, tmax=start + duration, return_times=True)
    data = data * 1e6
    data = data - data.mean(axis=1, keepdims=True)
    if spacing is None:
        spacing = 1.1 * np.max(np.percentile(data, 98, axis=1) - np.percentile(data, 2, axis=1))
    new_fig = ax is None
    if new_fig:
        fig, ax = plt.subplots(figsize=(11, 0.75 * len(channels) + 1.2))
    for i, ch in enumerate(channels):
        ax.plot(times, data[i] - i * spacing, lw=0.8, color="k")
    ax.set_yticks([-i * spacing for i in range(len(channels))], channels)
    ax.set_xlabel("Time (s)")
    ax.set_xlim(times[0], times[-1])
    # scale bar so you can read the size of the wiggles
    bar = float(f"{spacing:.1g}")
    x0 = times[-1] + 0.01 * duration
    ax.plot([x0, x0], [0, -bar], color="tab:red", lw=3, clip_on=False)
    ax.text(x0 + 0.01 * duration, -bar / 2, f"{bar:g} µV", color="tab:red", va="center")
    if title:
        ax.set_title(title)
    if new_fig:
        plt.tight_layout()
        plt.show()
    return spacing


def compare_traces(raws, channels, start=0, duration=10):
    """Same channels, same time window, one panel per version of the data."""
    fig, axes = plt.subplots(len(raws), 1, figsize=(11, len(raws) * (0.75 * len(channels) + 1)), sharex=True)
    spacing = None
    for ax, (label, raw) in zip(axes, raws.items()):
        spacing = show_traces(raw, channels, start, duration, title=label, spacing=spacing, ax=ax)
    plt.tight_layout()
    plt.show()


def compare_spectra(raws, fmax=80, picks=None, title=None):
    """Power at each frequency, averaged over channels, one line per version of the data."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, raw in raws.items():
        spectrum = raw.compute_psd(fmax=fmax, picks=picks, n_fft=int(2 * raw.info["sfreq"]))
        psd, freqs = spectrum.get_data(return_freqs=True)
        ax.plot(freqs, 10 * np.log10(psd.mean(axis=0) * 1e12), label=label, lw=1.2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_xlim(0, fmax)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(title or "Spectrum (average over channels)")
    plt.show()


def show_head(values, raw, title="", label="", cmap="Reds"):
    """Draw one number per electrode as a color map on the head (viewed from above, nose up)."""
    info = _good_eeg(raw, None).info
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap=cmap)
    plt.colorbar(im, ax=ax, label=label, shrink=0.8)
    ax.set_title(title)
    plt.show()


# Download everything up front so the rest of the notebook is instant
load_eegbci(4, 1), load_eegbci(4, 2), load_erp_core()
print("Ready.")
''', form=True)

# ----------------------------------------------------------------------------
md(r"""
---
## 1. Meet the data

The **PhysioNet EEG Motor Movement/Imagery dataset** has recordings from 109 volunteers wearing a 64-electrode EEG cap.
Before the main tasks, each volunteer sat still for **1 minute with eyes open** and then **1 minute with eyes closed**.

We'll start with **volunteer #4, eyes open**.
""")

code(r"""
raw = load_eegbci(subject=4, run=1)   # run 1 = sitting still, eyes open
raw                                    # show a summary of the recording
""")

md(r"""
### Predict (warm-up)
The cap has **64 electrodes**. Each one is measured **160 times per second**, for **61 seconds**.
How many numbers make up this recording?

**A)** ~10 thousand  **B)** ~600 thousand  **C)** ~60 million
""")

code(r"""
data = raw.get_data()            # pull out the raw numbers as one big table
print(data.shape)                # (number of electrodes, number of time points)
print(data.size, "numbers in total")
""")

answer(r"""
**B.** 64 electrodes × 9,760 time points = **624,640 numbers**.

That's all EEG is: **a table of voltages**, with one row per electrode and one column per moment in time.
Everything we do today is math on this table.
""")

md(r"""
### Where are the electrodes?
Electrode names follow the **10–20 system**:
- The **letter** is the brain region underneath: **Fp** = frontal pole (forehead), **F** = frontal, **C** = central, **P** = parietal, **T** = temporal, **O** = occipital (back of the head).
- The **number** is the side: **odd = left**, **even = right**, **z** = midline ("zero").
""")

code(r"""
raw.plot_sensors(show_names=True);   # map of the cap, seen from above (nose at the top)
""")

md(r"""
### Quick check
Without looking at the map: where is **O2**? Where is **C3**?
""")

answer(r"""
**O2** is at the back of the head, on the right. **C3** is on top, slightly left, over the **left motor cortex**, the part of the brain that moves your *right* hand.
""")

md(r"""
### Let's look at the signal
""")

code(r"""
show_traces(raw, ["Fp1", "Fz", "Cz", "Pz", "Oz"], start=20, duration=10)   # 10 seconds, front of head → back
""");

md(r"""
### Discuss
1. What do you notice? Anything that stands out?
2. Look at the red scale bar. How big are these signals?
3. Is this "brain waves"?
""")

answer(r"""
The signals are tiny: tens of **micro**volts (µV, millionths of a volt). An AA battery is 1.5 V, which is 1,500,000 µV.
Brain signals at the scalp are about **10,000 to 100,000 times smaller**, so almost anything electrical in the room can drown them out.

Did you spot the **big bump around 23.5 s**, biggest at Fp1 (the forehead)? Keep it in mind. We'll find out what it is in Section 6.

Is it brain waves? *We honestly can't tell yet.* That's the whole point of today.
""")

# ----------------------------------------------------------------------------
md(r"""
---
## 2. Voltage is always relative: the *reference*

**Voltage is a *difference*** in electrical potential between **two** points. A voltmeter always has two leads.

So an EEG electrode can't measure "the voltage at Cz" on its own. Every channel is really
**that electrode *minus* a reference electrode**.
""")

code(r"""
# Pretend we magically knew the "true" electrical potential at 4 spots (made-up units)
true_potential = {"Fp1": 12, "Cz": 7, "Oz": 3, "ear": 5}

reference = "ear"     # <- try changing this to "Cz" and re-running

for name, value in true_potential.items():
    print(f"{name:>4} reads {value - true_potential[reference]:+}")
""")

md(r"""
### Predict
Change `reference` to `"Cz"`.
1. What will **Cz** read?
2. Will the difference **Fp1 − Oz** change?
""")

answer(r"""
1. **Cz reads 0.** An electrode compared with itself is always 0.
2. **No.** Fp1 − Oz = 9 with either reference.

**Takeaways:**
- The reference shifts *every* channel's number, but never the differences *between* channels.
- **Whatever the reference electrode picks up shows up (flipped) in every channel.** A noisy reference means noise everywhere.
  That's why choosing the reference matters.
""")

md(r"""
### 2a. Common average reference (CAR)

Instead of trusting one electrode, **use the average of *all* electrodes as the reference.** At every moment we subtract the mean across electrodes from each one.

**Why?** Anything that hits *every* electrode equally (for example, noise picked up by the original reference) cancels out.
With lots of electrodes spread around the head, the average is a fairly neutral "zero".

**But first, a trap.** Look at these four neighboring electrodes on the right side of the head:
""")

code(r"""
show_traces(raw, ["FC6", "FT8", "T8", "C6"], start=20, duration=10)
""")

md(r"""
### Predict
1. Which electrode looks broken?
2. If we include it when we compute the average, what happens to the other 63 channels?
""")

answer(r"""
1. **FT8.** It swings far more than its neighbors, which sit only a few centimeters away. It probably had poor contact with the scalp.
2. Its junk gets **mixed into every single channel**, because every channel has the average subtracted from it.

So **we mark bad channels *before* re-referencing.**
""")

code(r"""
raw.info["bads"] = ["FT8"]                               # tell MNE: ignore this electrode from now on
raw_car = raw.copy().set_eeg_reference("average")        # subtract the average of all *good* electrodes
""")

md(r"""
### Predict
After CAR, if you **add up all the good channels** at any one moment, what do you get?
""")

code(r"""
car_data = raw_car.get_data(picks="eeg", exclude="bads")   # the good channels only
print(car_data.sum(axis=0)[:5])                            # sum across electrodes, for the first 5 moments
""")

answer(r"""
**Zero.** Numbers like `1e-20` are just computer rounding error.

If you subtract the average from every channel, the channels have to sum to zero. CAR literally defines "zero" as the head's average.
""")

code(r"""
compare_traces({"original reference": raw, "common average reference": raw_car},
               ["Fp1", "Cz", "Oz"], start=20, duration=10)
""")

md(r"""
### Discuss
Look at **Oz** (back of the head) at ~23.5 s in both panels. With the original reference the bump goes *up*. After CAR it goes **down**. Why would re-referencing flip it?
""")

answer(r"""
The bump is **huge at the front of the head**, so it drags the *average* of all electrodes upward.
CAR subtracts that average from every channel. At Oz, where the bump is small, subtracting a big average leaves a **negative copy**.

**CAR isn't magic.** It assumes nothing big is happening in just one spot. A large, localized signal leaks (flipped) into everyone else.
Every reference has trade-offs, and you'll see this one come back twice more today.
""")

md(r"""
### 2b. Bipolar montage

Another option: **subtract neighbors from each other.** "Fp1 − F3" asks *how different is Fp1 from the electrode just behind it?*

Anything that's the same at both electrodes (distant sources, shared noise) cancels out. What's left are **local differences**.
Hospital neurologists read clinical EEG this way, for example when looking for seizures.

Here's a chain running from the forehead to the back of the head, down the left side:
""")

md(r"""
### Predict
Remember the **big bump at ~23.5 s** at the front of the head? In which pair will it be **biggest**?

**A)** Fp1 − F3  **B)** F3 − C3  **C)** C3 − P3  **D)** P3 − O1
""")

code(r"""
chain = ["Fp1", "F3", "C3", "P3", "O1"]                                       # forehead → back of head
raw_bipolar = mne.set_bipolar_reference(raw, anode=chain[:-1], cathode=chain[1:])
show_traces(raw_bipolar, ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"], start=20, duration=10)
""")

answer(r"""
**A) Fp1 − F3.** The bump changes the most between the forehead and the electrode just behind it, and it barely differs between electrodes further back.

So whatever this thing is, it's **strongest at the very front of the head.** That's clue #1 for Section 6.

**Bonus:** around 26–29 s, **F3 − C3** and **C3 − P3** drift in *opposite* directions, like mirror images. Both pairs share **C3**, so C3 is the electrode doing something.
Neurologists call this a **phase reversal** and use it to pinpoint where a signal (like a seizure) is coming from.

*From here on we'll use the common average reference.*
""")

# ----------------------------------------------------------------------------
md(r"""
---
## 3. What's actually in this signal? Thinking in frequencies

The signal looks messy because it's a **mix of rhythms at many different speeds, plus noise.**

**Fourier's big idea:** any signal can be written as a sum of sine waves. It's like hearing a chord and picking out the individual notes.
The **Fourier transform (FFT)** does that for us: it tells us **how much of each frequency** is in the signal.

**Frequency** = cycles per second, measured in **Hertz (Hz)**. 10 Hz means 10 ups-and-downs every second.

Let's try it on a made-up signal first:
""")

code(r"""
sfreq = 160                                    # samples per second (same as our EEG)
t = np.arange(0, 2, 1 / sfreq)                 # 2 seconds of time points

signal = (1.0 * np.sin(2 * np.pi * 10 * t)     # a 10 Hz wave
          + 0.5 * np.sin(2 * np.pi * 25 * t)   # a smaller 25 Hz wave
          + 0.5 * np.random.randn(len(t)))     # random noise

plt.figure(figsize=(10, 3))
plt.plot(t, signal, color="k", lw=0.8)
plt.xlabel("Time (s)")
plt.title("A messy-looking signal")
plt.show()
""")

md(r"""
### Predict
We're about to take the FFT of this signal. **Sketch what you expect**: where will the peaks be, and how tall will they be compared with each other?
""")

code(r"""
freqs = np.fft.rfftfreq(len(signal), d=1 / sfreq)              # the frequencies the FFT checks
amplitude = np.abs(np.fft.rfft(signal)) * 2 / len(signal)      # how strong each frequency is

plt.figure(figsize=(10, 3))
plt.plot(freqs, amplitude, color="k")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("FFT of the messy signal")
plt.show()
""")

answer(r"""
There are **two peaks: one at 10 Hz (height ≈ 1) and one at 25 Hz (height ≈ 0.5)**, matching how we built the signal. The random noise is spread thinly across *all* frequencies and forms the low "floor".

**Something that looks messy in time can be simple in frequency.**
""")

md(r"""
### Now the real EEG
For real data we use the **power spectral density (PSD)**: chop the recording into short windows, FFT each one, and average.
The y-axis is on a **log scale (decibels)**, because brain signals span a huge range of strengths.

*Why does the x-axis stop at 80 Hz?* We sample 160 times per second, and you can only detect frequencies up to **half** your sampling rate.
""")

code(r"""
spectrum = raw_car.compute_psd(fmax=80)            # how much power at each frequency, for every channel
spectrum.plot(average=True, amplitude=False);      # average over channels (black line); grey = spread
""")

md(r"""
### Discuss
1. What's the overall shape?
2. There's one **sharp spike**. At what frequency? **What in this room oscillates at that frequency?**
""")

answer(r"""
1. Power is **highest at low frequencies and falls off** as frequency goes up. This "1/f" shape is typical of brain signals (and lots of natural signals).
2. The spike is at **60 Hz: power-line noise.** In the US, the electricity in the walls alternates **60 times per second**.
   That creates an oscillating electric field throughout the room, and your body and the electrode wires pick it up like an antenna.
   In Europe and much of Asia, the grid runs at **50 Hz**, so the spike would be there instead.

**It's not brain.** Let's remove it.
""")

# ----------------------------------------------------------------------------
md(r"""
---
## 4. Notch filter: delete one frequency

A **notch filter** removes a **narrow slice** of frequencies around one value and leaves everything else alone,
like noise-cancelling headphones tuned to a single hum.

### Predict
Sketch the spectrum **after** a 60 Hz notch filter.
""")

code(r"""
notch_freq = 60  #@param [50, 60] {type:"raw"}

raw_notch = raw_car.copy().notch_filter(freqs=notch_freq)    # cut out a narrow band around notch_freq
compare_spectra({"before notch": raw_car, "after notch": raw_notch})
""")

answer(r"""
The **spike is gone**. In its place there's a small *dip*, because the filter removes a little of everything in that narrow band, not just the noise.
**Everything else is unchanged.**

**Try it:** set `notch_freq` to **50** (the European line frequency) and re-run. The 60 Hz spike stays, and we've carved a hole at 50 Hz for nothing.
*You have to know where your noise is before you remove it.* Set it back to 60 before moving on.
""")

# ----------------------------------------------------------------------------
md(r"""
---
## 5. Band-pass filter: keep only the range we care about

Even without the hum, scalp EEG contains things we usually don't want:
- **Very slow drifts (below ~1 Hz):** sweat, electrodes shifting, breathing
- **Fast activity (above ~40 Hz):** mostly *muscle* (jaw, forehead, neck) plus electrical noise

A **band-pass filter** keeps only a range, here **1–40 Hz**. It's a *high-pass* (remove below 1 Hz) combined with a *low-pass* (remove above 40 Hz).

### Predict
What will the traces look like after a 1–40 Hz band-pass?

**A)** Smoother, and the slow wandering is gone  **B)** No visible change  **C)** A flat line
""")

code(r"""
low_cut = 1    #@param {type:"slider", min:0.1, max:20, step:0.1}
high_cut = 40  #@param {type:"slider", min:5, max:79, step:1}

raw_filt = raw_notch.copy().filter(l_freq=low_cut, h_freq=high_cut)    # keep only low_cut – high_cut Hz

compare_spectra({"original": raw_car, "notch": raw_notch, "notch + band-pass": raw_filt})
compare_traces({"before band-pass": raw_notch, "after band-pass": raw_filt},
               ["Fp2", "AF8", "F7"], start=0, duration=60)                     # the whole minute
""")

answer(r"""
**A.** The slow wandering (look at AF8 around 10–20 s) and the fast fuzz are gone, and the spectrum drops off outside 1–40 Hz.
The sharp spikes are still there, though. Hmm... (Section 6.)
""")

md(r"""
### Discuss
We low-passed at **40 Hz**. Did we even need the 60 Hz notch filter?
""")

answer(r"""
**Not for this analysis.** The 40 Hz low-pass already wipes out 60 Hz.

The notch matters when you want to **keep high frequencies** (for example "gamma" activity above 40 Hz) or when you aren't low-pass filtering.
Many pipelines do both just to be safe.
""")

md(r"""
### Filters can create what you're looking for
Try the sliders: set `low_cut = 8` and `high_cut = 12`, then re-run the cell above. Everything turns into smooth waves.
Are those real brain waves?

### Predict
We'll take **pure random noise**, with no brain involved, and keep only 8–12 Hz. What will it look like?
""")

code(r"""
noise = np.random.randn(5 * sfreq)                                      # 5 seconds of pure random noise
filtered_noise = mne.filter.filter_data(noise, sfreq, l_freq=8, h_freq=12)

fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
axes[0].plot(np.arange(len(noise)) / sfreq, noise, color="k", lw=0.8)
axes[0].set_title("Pure random noise")
axes[1].plot(np.arange(len(noise)) / sfreq, filtered_noise, color="k", lw=0.8)
axes[1].set_title("The same noise, band-passed to 8–12 Hz")
axes[1].set_xlabel("Time (s)")
plt.tight_layout()
plt.show()
""")

answer(r"""
Nice, regular **~10 Hz "brain waves"** out of nothing.
A narrow filter makes *anything* look rhythmic, so **seeing a wave after filtering is not evidence of a brain rhythm.**
To believe a rhythm is real, we need a **comparison**. That's what we'll do in Section 7.

*(Set the sliders back to 1 and 40 and re-run the band-pass cell before moving on.)*
""")

# ----------------------------------------------------------------------------
md(r"""
---
## 6. Noise filters can't fix: your eyes

Back to the mystery bump. Here's the band-passed data again:
""")

code(r"""
show_traces(raw_filt, ["Fp1", "Fp2", "F3", "Cz", "Pz", "Oz"], start=20, duration=10)
""")

md(r"""
### What is the big bump at ~23.5 s?
**A)** A thought  **B)** A blink  **C)** Power-line noise  **D)** A heartbeat

*Clues:* it's strongest at the forehead (Fp1/Fp2) and fades toward the back. It lasts a fraction of a second. What sits just below Fp1 and Fp2?
""")

md(r"""
### Checking with electrodes next to the eyes
To be sure, we'd want electrodes placed right around the eyes. These are called **EOG** (electro-**oculo**-gram).
Our dataset doesn't have any, so we'll borrow a recording from a different lab (**ERP CORE**, UC Davis) that has electrodes **above** the eye (FP2) and **below** it (VEOG_lower).

**Physics:** the eyeball works like a tiny battery. The front (cornea) is electrically positive compared with the back (retina).
When you blink, the eyelid slides over the eye and the eye rolls slightly, which shifts that charge. Electrodes near the eye see a big change.

### Predict
During a blink, how will the electrode **above** the eye compare with the one **below** it?

**A)** Identical  **B)** Mirror images (opposite directions)  **C)** Only the one above changes
""")

code(r"""
raw_eog = load_erp_core()                                     # a different person, a different lab
raw_eog.filter(l_freq=1, h_freq=40, picks=["eeg", "eog"])     # same band-pass as before

show_traces(raw_eog, ["FP2", "VEOG_lower", "Oz"], start=10, duration=10)
""")

answer(r"""
**B) Mirror images.** At each blink, the electrode above the eye swings one way and the one below swings the *opposite* way, because they sit on opposite sides of the eye's "battery".
**Oz**, at the back of the head, barely notices.

So the bump in our data is **a blink**: biggest at the forehead, mirrored across the eye, and absent at the back.
""")

md(r"""
### Predict: bipolar montage, again
What happens to the blink if we compute **above-eye − below-eye** (a bipolar pair)?

**A)** It gets **bigger**  **B)** It **cancels out**
""")

code(r"""
raw_eog = mne.set_bipolar_reference(raw_eog, anode="FP2", cathode="VEOG_lower", ch_name="VEOG", drop_refs=False)
show_traces(raw_eog, ["FP2", "VEOG_lower", "VEOG"], start=10, duration=10)
""")

answer(r"""
**A) Bigger.** Subtracting a mirror image *adds* it: (+100) − (−100) = 200.
Meanwhile, brain activity that's about the same at both electrodes cancels out.
That's exactly why eye channels are recorded as **bipolar pairs**: they make the eyes stand out.
""")

md(r"""
### Where on the head does a blink show up?
Back to our volunteer. MNE can find the blinks in Fp1 automatically, cut out a window around each one, and average them.
""")

md(r"""
### Predict
Picture a map of the head (nose at the top). **Where will the color be strongest** at the moment of the blink?
""")

code(r"""
blinks = mne.preprocessing.create_eog_epochs(raw_filt, ch_name="Fp1")    # find blinks using Fp1, cut out each one
print(len(blinks), "blinks found")
blinks.average().plot_topomap(times=0, ch_type="eeg", size=3);           # head map at the peak of the average blink
""")

answer(r"""
**At the front, over the eyes**, fading toward the back. That's the signature of an eye artifact.

*Why is the back of the head blue (negative)?* That's CAR again: the huge frontal blink pulls the average up, so the back gets a flipped copy.
""")

md(r"""
### Why not just filter out blinks?
A blink lasts about a third of a second, so most of its power is around **1–5 Hz**. That's the **same range as real brain rhythms** (delta and theta).
A filter can't separate them, and removing those frequencies would delete brain signal too.

Real pipelines use smarter tools instead:
- **Throw out** the seconds that contain blinks
- **Regress out** the EOG channel
- **ICA** (independent component analysis): un-mix the signal into sources and delete the "eye" source. *(Future meeting!)*

Eyes aren't the only culprit. Common artifacts include:

| Artifact | Looks like |
|---|---|
| Blinks / eye movements | big slow bumps at the front |
| Jaw clench, frowning | bursts of fast fuzz (> 20 Hz), often at the temples |
| Heartbeat | small regular spikes about once a second |
| Moving or sweating | slow drifts, sudden jumps |
| Power lines | 50/60 Hz hum |

### Discuss
Consumer "focus-tracking" headbands put their electrodes on the **forehead**. What might they *actually* be measuring?
""")

answer(r"""
A lot of **eye movements, blinks, and forehead muscle tension.** Those do change when you concentrate, but they aren't brain activity.
Good products try to separate them out, but it's always worth asking: *is this really brain?*
""")

# ----------------------------------------------------------------------------
md(r"""
---
## 7. Is any of this actually brain? Eyes open vs. eyes closed

In **1929**, Hans Berger published the first recordings of human EEG. He saw a big rhythm, roughly 10 per second,
that **appears when you close your eyes and disappears when you open them**. He called it the **alpha rhythm**.
Let's see if we can reproduce one of the oldest results in neuroscience.

### Explain the recipe
Here's our whole cleaning pipeline applied to the **eyes-closed** recording.
**Before running it, explain to your table what each line does and *why*.**
""")

code(r"""
raw_closed = load_eegbci(subject=4, run=2)                          # run 2 = sitting still, eyes CLOSED
raw_closed.info["bads"] = ["FT8"]
raw_closed.set_eeg_reference("average")
raw_closed.notch_filter(freqs=60)
raw_closed.filter(l_freq=1, h_freq=40)

raw_open = raw_filt                                                  # the eyes-open data we cleaned above
""")

answer(r"""
1. Load the eyes-closed recording.
2. Mark the broken electrode as bad so it doesn't contaminate anything.
3. Re-reference to the common average.
4. Remove the 60 Hz power-line hum.
5. Keep only 1–40 Hz (drop the slow drifts and fast muscle noise).

These are *exactly* the steps we used for the eyes-open data, which is important for a fair comparison.
""")

md(r"""
### Predict
1. Which part of the head will change most when the eyes close?
2. At what frequency?
""")

code(r"""
compare_traces({"eyes OPEN": raw_open, "eyes CLOSED": raw_closed},
               ["Fz", "Cz", "Pz", "O1", "Oz", "O2"], start=20, duration=10)
""")

md(r"""
### Count the waves
Here's **one second** of the eyes-closed recording at the back of the head. Count the peaks.
""")

code(r"""
show_traces(raw_closed, ["O1", "Oz", "O2"], start=24, duration=1)
""")

code(r"""
compare_spectra({"eyes open": raw_open, "eyes closed": raw_closed},
                fmax=40, picks=["O1", "Oz", "O2"], title="Back of the head (O1, Oz, O2)")
""")

md(r"""
### Predict: on the head
We'll compute **how many times more 8–12 Hz power** there is with eyes closed than open, at every electrode, and paint it on the head.
Where will it be reddest?
""")

code(r"""
alpha_open = raw_open.compute_psd(fmin=8, fmax=12).get_data().mean(axis=1)      # alpha power per electrode, eyes open
alpha_closed = raw_closed.compute_psd(fmin=8, fmax=12).get_data().mean(axis=1)  # ... and eyes closed

show_head(alpha_closed / alpha_open, raw_open,
          title="Alpha power: eyes closed ÷ eyes open", label="× more alpha")
""")

answer(r"""
- The big, regular rhythm is at about **10 waves per second** (the spectrum peaks near **10–11 Hz**). That's alpha.
- It's strongest at the **back of the head**, over the **occipital lobe: the visual cortex.** When your eyes close, visual cortex stops processing input and its neurons fall into a synchronized ~10 Hz rhythm.

**Why we believe this one is real brain activity** (unlike the 60 Hz hum, the blinks, or the filtered noise):
- It changes with **behavior**: eyes open vs. closed.
- It sits over the **right anatomy**: the visual cortex.
- We cleaned both recordings **identically**, so the filters can't explain the difference.
- It **replicates** a nearly 100-year-old finding.

*Notice Fz also wiggles a bit with eyes closed?* Part of that is CAR one more time: the big occipital alpha is part of the average, so a flipped copy leaks to the front of the head.
""")

# ----------------------------------------------------------------------------
md(r"""
---
## 8. Recap

| Step | Code | What it does | Why |
|---|---|---|---|
| Load | `load_eegbci(...)` | Reads a table of voltages (electrodes × time) | That's all EEG is |
| Mark bad channels | `raw.info["bads"] = [...]` | Ignores broken electrodes | One bad channel can contaminate the rest |
| Re-reference | `set_eeg_reference("average")` | Measures each channel relative to the head's average | Voltage is always relative; choose a sensible zero |
| Notch filter | `notch_filter(freqs=60)` | Deletes a narrow band at 60 Hz | Power lines in the walls |
| Band-pass | `filter(l_freq=1, h_freq=40)` | Keeps 1–40 Hz | Removes drifts and muscle noise |
| Check artifacts | look at traces & head maps | Spots blinks and other non-brain signals | Filters can't remove everything |
| Compare conditions | `compute_psd(...)` | Eyes open vs. closed | The only way to be sure a pattern is real |

### Big ideas
1. **EEG is a table of voltages, and voltage is always relative** to a reference.
2. **Most of what you see in raw EEG isn't brain**: power lines, drifts, muscles, eyes.
3. **The frequency view** makes some noise easy to spot and remove. Other noise (blinks) overlaps brain frequencies and needs smarter tools.
4. **Filters can manufacture patterns.** Trust a result when it shows up as a *difference between conditions*.

### Next time
The same dataset has runs where volunteers **imagined moving their left or right hand.** Can a computer tell which one they imagined, just from EEG?
That's a **brain–computer interface**.

---
**Data & tools:**
[PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) (Schalk et al., 2004, *IEEE TBME*; Goldberger et al., 2000, *Circulation*) ·
[ERP CORE](https://erpinfo.org/erp-core) (Kappenman et al., 2021, *NeuroImage*; CC BY-SA 4.0) ·
[MNE-Python](https://mne.tools) (Gramfort et al., 2013, *Frontiers in Neuroscience*)
""")

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"provenance": [], "toc_visible": True},
}
nbf.write(nb, sys.argv[1])
print("wrote", sys.argv[1], len(cells), "cells")
