import os
import sys
import json
import shutil
import logging
import datetime as dt

import pydub
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pandas.plotting import register_matplotlib_converters
from pydub import AudioSegment
import speech_recognition as sr
from wordcloud import WordCloud
from speechToCloud import SpeechCloud

register_matplotlib_converters()
logger = logging.getLogger(__name__)


# HELPER FUNCTIONS:
def splitAndSaveAudio(srcFile, tgtDir, chunkDurationMin, tgtFormat="wav"):
    """
    use pydub to split and save original mp3 file into chunks
    of given duration.

    Args:
        srcFile: (str) full path to audio file
        tgtDir: (str) path to target directory where chunks will be stored
        chunkDurationMin: (int or float) desired chunk duration
        tgtFormat: (str) default is "wav"

    Returns:

    """

    _, file = os.path.split(srcFile)
    fileName, fileFormat = file.split(".")
    audiofile = AudioSegment.from_file(srcFile, format=fileFormat)
    durationMs = len(audiofile)

    oneMinMs = 60 * 1000  # pydub works in milliseconds
    chunkDurationMs = chunkDurationMin * oneMinMs

    counter = 1
    startMs = 0
    stopMs = startMs + chunkDurationMs
    while startMs <= durationMs:

        if stopMs > durationMs:
            stopMs = durationMs

        chunkFile = f"{fileName}_{counter}.{tgtFormat}"
        chunk = audiofile[startMs:stopMs]
        chunk.export(os.path.join(tgtDir, chunkFile), format=tgtFormat)
        del chunk

        startMs = stopMs + 1
        stopMs = startMs + chunkDurationMs
        counter += 1

    del audiofile
    print(f"splitAndSaveAudio - {file}: DONE")

def loadMp3AsArray(filePath):
    """
    load mp3 as ndarray using librosa.

    Args:
        filePath:

    Returns:
        x: mp3 as ndarray (nSamples, )
        sr: sampling resolution

    """
    x, sr = librosa.load(filePath, sr=None)

    return x, sr

def saveJsonSegments(tgtFile, segments):

    if os.path.exists(tgtFile):
        os.remove(tgtFile)

    with open(tgtFile, "w") as f:
        json.dump(segments, f)

def saveMp3Segments(srcFile, tgtDir, segments):

    parentDir, mp3File = os.path.split(srcFile)
    chunksDir = os.path.join(tgtDir, mp3File.split(".")[0])

    # if folder already exists, remove and replace
    if os.path.exists(chunksDir):
        shutil.rmtree(chunksDir)
        os.makedirs(chunksDir)
    else:
        os.makedirs(chunksDir)

    print("loading: " + mp3File)
    audiofile = pydub.AudioSegment.from_mp3(srcFile)

    songTimes = segments["songTimes"]
    for i, segSong in enumerate(songTimes):
        chunkFile = f"song_{i}.mp3"
        startMs = segSong[0] * 1000
        stopMs = segSong[1] * 1000
        print("creating: " + chunkFile + " ...")
        chunk = audiofile[startMs:stopMs]
        chunk.export(os.path.join(chunksDir, chunkFile), format="mp3")
        del chunk

    noSongTimes = segments["noSongTimes"]
    for k, segNoSong in enumerate(noSongTimes):
        chunkFile = f"noSong_{k}.mp3"
        startMs = segNoSong[0] * 1000
        stopMs = segNoSong[1] * 1000
        print("creating: " + chunkFile + " ...")
        chunk = audiofile[startMs:stopMs]
        chunk.export(os.path.join(chunksDir, chunkFile), format="mp3")
        del chunk

    print(f"{mp3File} all segments exported!")


# SPEECH-VS-SONG SEGMENTATION:
def getMfccDict(y, sr, hopLength=512, numFft=2048, numMfcc=20, show=False):
    """

    Args:
        y:
        sr:
        hopLength:
        numFft:
        numMfcc:

    Returns:

    """
    mfccDict = dict()
    mfccDict["mfccs"] = librosa.feature.mfcc(
        y=y, sr=sr, hop_length=hopLength, n_fft=numFft, n_mfcc=numMfcc
    )
    mfccDict["mfccFrames"] = np.array(np.arange(mfccDict["mfccs"].shape[1]))
    mfccDict["mfccTimes"] = librosa.frames_to_time(
        frames=mfccDict["mfccFrames"], sr=sr, hop_length=hopLength
    )

    if show:

        plt.figure()
        plt.subplot(2, 2, 1)
        plt.plot(y)
        plt.xlim(0, len(y))

        ax2 = plt.subplot(2, 2, 3)
        imgMfcc = librosa.display.specshow(
            mfccDict["mfccs"], sr=sr, hop_length=hopLength, x_axis="time", ax=ax2
        )
        ax2.set_ylabel("MFCC")

    return mfccDict

def getSmoothedFirstMfcc(mfccDict, smoothOverNseconds=60):
    """

    Args:
        mfccDict: (dict) output of getMfccDict
        smoothOverNseconds: (int) number of seconds to smooth over

    Returns:
        smoothedSer: (pd.Series), with index in seconds

    """

    # explore the first MFCC:
    mfccTimes = mfccDict["mfccTimes"]
    oneSecFrames = int(1 / np.median(np.diff(mfccTimes)))
    smoothOverNframes = oneSecFrames * smoothOverNseconds
    firstMfcc = mfccDict["mfccs"][0, :]
    ser = pd.Series(firstMfcc, index=mfccTimes)
    smoothedSer = ser.rolling(window=smoothOverNframes, center=True).median()

    return smoothedSer

def exploreFirstMfcc(dirPath, thrFirstMfcc=None):
    """

    Args:
        dirPath:

    Returns:

    """

    files = [
        file
        for file in os.listdir(dirPath)
        if os.path.isfile(os.path.join(dirPath, file))
    ]

    firstMfccDir = os.path.join(dirPath, "firstMfccs")
    if not os.path.exists(firstMfccDir):
        os.makedirs(firstMfccDir)

    for file in files:

        print(f"processing file: {file} ...")

        fName = file.split(".")[0]
        fp = os.path.join(dirPath, file)
        y, sr = loadMp3AsArray(filePath=fp)
        mfccDict = getMfccDict(
            y, sr, hopLength=512, numFft=2048, numMfcc=20, show=False
        )
        firstMfcc = mfccDict["mfccs"][0, :]
        smoothedFirstMfcc = getSmoothedFirstMfcc(mfccDict, smoothOverNseconds=60)

        mfccMean = smoothedFirstMfcc.mean()
        mfccMedian = smoothedFirstMfcc.median()
        mfccMode = stats.mode(smoothedFirstMfcc)[0]

        # if thrFirstMfcc is None:
        #     thrFirstMfcc = mfccMode + 1 * np.std(smoothedFirstMfcc)

        # # smoothedPctChange = smoothedFirstMfcc.pct_change().rolling(window=smoothOverNframes, center=True).median()
        # aux = smoothedFirstMfcc
        # aux[np.isnan(aux)] = 0.0
        # detrendedSmoothed = signal.detrend(aux)
        # reDetrendedSmoothed = detrendedSmoothed - stats.mode(detrendedSmoothed)[0]

        fig = plt.figure(figsize=(19.0, 10.0))
        plt.subplot(221)
        plt.plot(mfccDict["mfccTimes"], firstMfcc, color="lightgrey")
        plt.plot(mfccDict["mfccTimes"], smoothedFirstMfcc, color="orange")
        # plt.legend(["original", "smoothed over 1 min"])
        plt.ylabel("coefficients"), plt.xlabel("time (seconds)")
        plt.title(f"{file}: first MFCC")
        plt.axhline(mfccMean, linestyle="--", color="r")
        plt.axhline(mfccMedian, linestyle=":", color="m")
        plt.legend(["original", "smoothed over 1 min", "mean", "median"])
        if thrFirstMfcc:
            plt.axhline(thrFirstMfcc, linestyle="--", color="k")

        plt.subplot(222)
        plt.hist(smoothedFirstMfcc, bins=50, color="orange")
        plt.axvline(mfccMean, linestyle="--", color="r")
        plt.axvline(mfccMedian, linestyle="--", color="m")
        if thrFirstMfcc:
            plt.axvline(thrFirstMfcc, linestyle="--", color="k")
        plt.ylabel("count"), plt.xlabel("smoothed first MFCC")
        plt.title(f"{file}: distribution")

        # plt.subplot(223)
        # # plt.plot(mfccDict['mfccTimes'], smoothedPctChange)
        # plt.plot(mfccDict['mfccTimes'], smoothedFirstMfcc)
        # plt.plot(mfccDict['mfccTimes'], detrendedSmoothed)
        # plt.plot(mfccDict['mfccTimes'], reDetrendedSmoothed)
        # plt.ylabel("detrended"), plt.xlabel("time (seconds)")

        figName = fName + ".png"
        fig.canvas.start_event_loop(sys.float_info.min)
        plt.savefig(os.path.join(firstMfccDir, figName), bbox_inches="tight")

        plt.close()

        del y, sr, mfccDict, firstMfcc, smoothedFirstMfcc

    print("exploreFirstMfcc: FINISHED")

def getSegmentTimes(y, sr, minSongDuration=60, thrMethod="mean", figurePath=None):
    """

    "song" segments correspond to intervals with smoothed MFCC larger then the mean for
    at least 1 consecutive minute. By contrast, "no song" segments are those between
    detected "song" ones.

    Args:
        y:
        sr:
        minSongDuration:
        thrMethod: (str) 'mean' [default] or 'median'
        figurePath:

    Returns:
        segments: (dict) storing 'song' and 'noSong' tuples with the (start,stop) seconds of each segment

    """

    mfccDict = getMfccDict(y, sr, hopLength=512, numFft=2048, numMfcc=20)
    mfccTimes = mfccDict["mfccTimes"]
    firstMfcc = mfccDict["mfccs"][0, :]
    oneSecFrames = int(1 / np.median(np.diff(mfccTimes)))
    smoothedFirstMfcc = getSmoothedFirstMfcc(mfccDict, smoothOverNseconds=60)

    if thrMethod == "mean":
        thrFirstMfcc = smoothedFirstMfcc.mean()
    elif thrMethod == "median":
        thrFirstMfcc = smoothedFirstMfcc.median()
    else:
        thrFirstMfcc = smoothedFirstMfcc.mean()
        print(f"'{thrMethod}' is not a recognized method. Using 'mean' instead")

    songCandidates = smoothedFirstMfcc > thrFirstMfcc
    minSongDurationFrames = int(minSongDuration * oneSecFrames)

    # find times of song segments:
    songTimes = []
    firstFrame = 0
    while firstFrame < len(songCandidates):

        # if frame considered song candidate:
        if songCandidates.iloc[firstFrame]:
            lastFrame = firstFrame + minSongDurationFrames

            if lastFrame < len(songCandidates):
                candidateChunk = songCandidates.iloc[firstFrame:lastFrame]

                # if true for at least one consecutive minute:
                if sum(candidateChunk) == len(candidateChunk):
                    songStartTime = mfccTimes[firstFrame]
                    firstFrame = lastFrame
                    while (firstFrame + 1 < len(songCandidates)) and (
                        songCandidates.iloc[firstFrame + 1]
                    ):
                        firstFrame += 1
                    songStopTime = mfccTimes[firstFrame]
                    songTimes.append((songStartTime, songStopTime))

                    firstFrame += 1

                else:
                    firstFrame += 1

        # if frame not song-candidate:
        else:
            firstFrame += 1

    # define times of no-song segments as the intervals between songs
    noSongTimes = []
    for i in range(len(songTimes)):

        if i == 0:
            noSongStartTime = 0
            noSongStopTime = songTimes[i][0] - 1
            noSongTimes.append((noSongStartTime, noSongStopTime))

        elif i == len(songTimes) - 1:
            noSongStartTime = songTimes[i][1] + 1
            noSongStopTime = songCandidates.index[-1]
            noSongTimes.append((noSongStartTime, noSongStopTime))

        else:
            noSongStartTime = songTimes[i - 1][1] + 1
            noSongStopTime = songTimes[i][0] - 1
            noSongTimes.append((noSongStartTime, noSongStopTime))

    segments = dict()
    segments["songTimes"] = songTimes
    segments["noSongTimes"] = noSongTimes

    if figurePath:

        _, file = os.path.split(figurePath)

        fig = plt.figure(figsize=(19.0, 10.0))
        plt.subplot(221)
        plt.plot(mfccDict["mfccTimes"], firstMfcc, color="lightgrey")
        plt.plot(mfccDict["mfccTimes"], smoothedFirstMfcc, color="orange")
        plt.axhline(thrFirstMfcc, linestyle="--", color="k")
        plt.ylabel("coefficients"), plt.xlabel("time (seconds)")
        plt.title(f"{file}: first MFCC")
        plt.legend(["original", "smoothed over 1m", f"{thrMethod}"])
        for i in segments["noSongTimes"]:
            plt.axvline(i[0], linestyle="-", color="cornflowerblue", lw=0.5)
            plt.axvline(i[1], linestyle="-", color="cornflowerblue", lw=0.5)
        for i in segments["songTimes"]:
            plt.axvline(i[0], linestyle="-", color="red")
            plt.axvline(i[1], linestyle="-", color="red")

        plt.subplot(222)
        plt.hist(smoothedFirstMfcc, bins=100, color="orange")
        plt.axvline(thrFirstMfcc, linestyle="--", color="k")
        plt.ylabel("count"), plt.xlabel("smoothed first MFCC")
        plt.title(f"{file}: distribution")

        fig.canvas.start_event_loop(sys.float_info.min)
        plt.savefig(figurePath, bbox_inches="tight")
        plt.close(fig)

    return segments

def doSegmentation(lstFileSrc):

    import time

    t0 = time.process_time()
    for filePath in lstFileSrc:

        dirPath, file = os.path.split(filePath)

        print(f"processing file: {file} ...")

        y, sr = loadMp3AsArray(filePath)
        figPath = os.path.join(dirPath, "segmentsPng", file[:-4] + ".png")
        segments = getSegmentTimes(
            y, sr, minSongDuration=60, thrMethod="mean", figurePath=figPath
        )
        jsonPath = os.path.join(dirPath, "segmentsJson", file[:-4] + ".json")
        saveJsonSegments(tgtFile=jsonPath, segments=segments)
        saveMp3Segments(
            filePath, tgtDir=os.path.join(dirPath, "segmentsMp3"), segments=segments
        )

    t1 = time.process_time()
    print("Elapsed time {}".format(time.strftime("%H:%M:%S", time.gmtime(t1 - t0))))

def plotAudienceBySegment(lstFileSrc):

    startTime = dt.time(21, 00, 00)
    srcPath = r"P:\WORK\PYTHONPATH\RUG\projects\autoradiolockdown\ruggero-dev"
    dirPath = os.path.join(srcPath, r"autolog\audio\puntate")
    jsonPath = os.path.join(dirPath, "mp3", "segmentsJson")
    audiencePath = os.path.join(srcPath, r"audience\scrapedData\xls\puntatePassate")
    pngPath = os.path.join(dirPath, "png")
    audienceXls = os.listdir(audiencePath)

    for filePath in lstFileSrc:

        _, file = os.path.split(filePath)
        puntata = file[:-4]
        print(f"plotAudienceBySegment: {puntata} ...")

        try:
            audienceFile = [x for x in audienceXls if x.startswith(puntata)][0]
            print(f"plotAudienceBySegment: {puntata} ...")
            df = pd.read_excel(os.path.join(audiencePath, audienceFile))
            df["song"] = False
            df["speech"] = False

            with open(os.path.join(jsonPath, puntata + ".json")) as f:
                segments = json.load(f)

            sessionDate = df["datetime"][0].strftime("%a-%d-%b-%Y")
            day = df["datetime"][0].to_pydatetime().date()
            startDateTime = dt.datetime.combine(day, startTime)

            totDuration = {
                "song": dt.timedelta(seconds=0),
                "speech": dt.timedelta(seconds=0),
                "songMean": np.nan,
                "speechMean": np.nan,
            }
            for k in segments.keys():
                logType = "song" if k.startswith("song") else "speech"
                segTimes = segments[k]
                for times in segTimes:
                    start = startDateTime + dt.timedelta(seconds=int(times[0]))
                    stop = startDateTime + dt.timedelta(seconds=int(times[1]))
                    duration = stop - start
                    totDuration[logType] += duration

                    print(
                        f"{logType} | start: {start.time()}, stop: {stop.time()}, duration: {duration}, totDuration: {totDuration[logType]}"
                    )

                    # plot only if segment duration > 30 seconds:
                    if duration >= dt.timedelta(seconds=30):

                        # TODO: CHECK THAT THIS IS WORKING AS IT SHOULD
                        iStart = df["datetime"].apply(lambda x: abs(x - start)).idxmin()
                        iStop = df["datetime"].apply(lambda x: abs(x - stop)).idxmin()
                        df.loc[iStart:iStop, logType] = True
                        # df[logType].iloc[iStart:iStop+1] = True

                meanDuration = totDuration[logType] / len(segTimes)
                totDuration[logType + "Mean"] = meanDuration - dt.timedelta(
                    microseconds=meanDuration.microseconds
                )

            # plotting variables:
            smoothOverMins = 10
            scrapeEverySecs = np.median(
                np.diff(df["datetime"]) / np.timedelta64(1, "s")
            )
            scrapesPerMin = int(60 / scrapeEverySecs)
            winLength = smoothOverMins * scrapesPerMin
            smoothed = df["current"].rolling(window=winLength, center=True).median()
            smoothedSong = smoothed.copy()
            smoothedSong[~df["song"]] = np.nan
            smoothedSpeech = smoothed.copy()
            smoothedSpeech[~df["speech"]] = np.nan

            # plot:
            songColor = "gold"
            speechColor = "cornflowerblue"
            fig = plt.figure(figsize=(19.0, 10.0))
            plt.title(puntata)
            plt.fill_between(
                df["datetime"], df["current"], color="cornflowerblue", alpha=0.2
            )
            plt.plot(df["datetime"], smoothedSpeech, color=speechColor, lw=3)
            plt.plot(df["datetime"], smoothedSong, color=songColor, lw=3)
            plt.legend(
                [
                    f"speech (tot duration: {totDuration['speech']}, mean: {totDuration['speechMean']})",
                    f"song (tot duration: {totDuration['song']}, mean: {totDuration['songMean']})",
                ]
            )
            plt.ylabel("Ascoltatori", fontsize=12)
            plt.grid(axis="y", ls=":"), plt.ylim([0, np.nanmax(df["current"]) + 5])
            plt.title("Radio Lockdown: {}".format(sessionDate), fontweight="bold")

            # save:
            figName = f"segmented-{puntata}.png"
            fig.canvas.start_event_loop(sys.float_info.min)
            plt.savefig(os.path.join(pngPath, figName), bbox_inches="tight")
            plt.close()

        except:
            print(f"{puntata} has no audience-tracking file, skipped ...")


# SPEECH-TO-TEXT:
def speechToText(wavFile, language="it-IT", engine="googleCloudAPI"):
    """

    Args:
        wavFile: (str) full path to wav audio file
        language: (str)
        engine: (str) either "googleCloudAPI" [default] or "google"

    Returns:
        text

    """

    recog = sr.Recognizer()
    recog.operation_timeout = 120

    with sr.AudioFile(wavFile) as source:
        recog.adjust_for_ambient_noise(source)
        audio = recog.record(source)

    try:

        if engine == "google":
            text = recog.recognize_google(audio, language=language)
            
        elif engine == "googleCloudAPI":

            with open(r"P:\WORK\PYTHONPATH\RUG\docs\radiolockdown-0961f41fa3dc.json") as f:
                credentialsGoogleCloudAPI = f.read()

            text = recog.recognize_google_cloud(audio, language=language, credentials_json=credentialsGoogleCloudAPI)
            
        else:
            print(f"'{engine}' is not a recognized speech-recognition engine. returning empty text")
            text = ""

        return text

    except Exception as e:
        file = os.path.split(wavFile)[1]
        print(f"speechToText on: {file} not possible, returning ''\n --> {e}")
        emptyString = ""
        return emptyString

def textFromMultipleSpeechFiles(wavDir, language="it-IT", engine="googleCloudAPI"):
    """

    Args:
        wavDir: (str) full path to directory containing wav audio files (each smaller then 10 MB!!!)
        language:
        engine: (str) either "googleCloudAPI" [default] or "google"

    Returns:
        txt: (str)

    """

    longTxt = []

    for wav in os.listdir(wavDir):
        wavFilePath = os.path.join(wavDir, wav)

        try:
            txt = speechToText(wavFilePath, language=language, engine=engine)

        except:
            print(f"{wav}: didn't work ... skipping it")
            txt = ""

        longTxt.append(" " + txt)

    return " ".join(longTxt)

def getTranscriptionFromSegments(podcast, tgtDir=None):
    """
    Important: before running this function, run 'doSegmentation' to segment the continuous mp3 of the podcast

    Args:
        podcast: (str or list of str) with the name of the podcasts (e.g. 'puntata1', 'puntata20', ...)
        tgtDir: (str) path to folder where to store the transcriptions

    Returns:
        save txt files

    """

    if isinstance(podcast, str):
        podcast = [podcast]

    src = r"P:\WORK\PYTHONPATH\RUG\projects\autoradiolockdown\ruggero-dev\autolog\audio\puntate"
    dirMp3Segments = os.path.join(src, r"mp3\segmentsMp3")

    if tgtDir is None:
        dirTxt = os.path.join(src, "txt")
    else:
        dirTxt = tgtDir

    for episode in podcast:

        dirSegmentsPuntata = os.path.join(dirMp3Segments, episode)
        dirTxtPuntata = os.path.join(dirTxt, episode)

        speechFiles = [x for x in os.listdir(dirSegmentsPuntata) if x.startswith("noSong") and x.endswith("mp3")]

        for mp3FileName in speechFiles:

            print(f"speech to text: {episode}, {mp3FileName}")

            mp3filePath = os.path.join(dirSegmentsPuntata, mp3FileName)
            fileName = mp3FileName.split(".")[0]

            # (create) temporary folder to store wav chunks:
            chunksDir = os.path.join(dirSegmentsPuntata, fileName + "_chunks")
            if not os.path.exists(chunksDir):
                os.makedirs(chunksDir)

            try:
                splitAndSaveAudio(srcFile=mp3filePath, tgtDir=chunksDir, chunkDurationMin=1, tgtFormat="wav")
                txt = textFromMultipleSpeechFiles(chunksDir, language="it-IT", engine="googleCloudAPI")
            except:
                txt = ""

            # (create) folder to store recognized txt for each speech segment of each episode:
            if not os.path.exists(dirTxtPuntata):
                os.makedirs(dirTxtPuntata)

            # write txt file with the recognized speech
            with open(os.path.join(dirTxtPuntata, fileName + ".txt"), "w+") as txtFile:
                txtFile.write(txt)


        # delete all temporary wav folders:
        wavFoldersList = [x for x in os.listdir(dirSegmentsPuntata) if x.endswith("chunks")]
        for wavFolder in wavFoldersList:
            shutil.rmtree(os.path.join(dirSegmentsPuntata, wavFolder))


        # merge all txt files of different speech segments together (summary di tutta la episode):
        txtSegments = [x for x in os.listdir(dirTxtPuntata) if x.startswith("noSong") and x.endswith("txt")]
        textList = []
        for file in txtSegments:
            txtFilePath = os.path.join(dirTxtPuntata, file)
            with open(txtFilePath, "r") as txtFile:
                txt = txtFile.read()
            textList.append(txt)
        wholeText = " ".join(textList)

        # and save to txt file:
        with open(os.path.join(dirTxtPuntata, episode + ".txt"), "w+") as txtFile:
            txtFile.write(wholeText)

        print(f"getTranscriptionFromSegments: Finished {episode}")

    print("getTranscriptionFromSegments: FINISHED")



# WORD CLOUD:
def getStopWords(text, language='it', stopList=None):
    """
    Remove every word that is neither a NOUN or a PROPN

    Args:
        text: (str)
        language: (str) default is 'it'
        stopList: (list of str) of additional stop words

    Returns:
        stopWords: (list of str)

    """

    import spacy
    from stop_words import get_stop_words

    stopWordsDict = {
        "it": ["c'è", "parte", "l'ha"]
    }

    if stopList is None:
        stopList = stopWordsDict[language]
    else:
        stopList = stopWordsDict[language] + stopList

    if language == 'it':
        nlp = spacy.load('it_core_news_lg')
        stopWords = get_stop_words('it') + stopList
    else:
        print(f"{language} still not implemented n this function, returning empty list")
        return []

    doc = nlp(text)

    for token in doc:

        notNounOrProp = (token.pos_ != 'NOUN') and (token.pos_ != 'PROPN')

        if notNounOrProp:
            stopWords.append(token.text)

    return stopWords

def plotWordCloud(text, maxWords=200, stopwords=None, backgroundColor="white", colormap="viridis"):
    """
    Plot Wordcloud

    Args:
        text: text to convert to wordcloud
        maxWords: max num of words to plot
        stopwords: list of str, words to remove from the computation
        backgroundColor: str [default = "white"]
        colormap: matplotlib colormap [default = "viridis"]

    Returns:
        fig handle

    """

    if stopwords is not None and not isinstance(stopwords, set):
        stopwords = set(stopwords)

    fig = plt.figure(figsize=(19.0, 19.0))

    try:
        wordcloud = WordCloud(
            max_words=maxWords,
            stopwords=stopwords,
            background_color=backgroundColor,
            colormap=colormap,
            width=3000,
            height=3000,
            random_state=1,
            collocations=True
        ).generate(text)

        plt.imshow(wordcloud)
        plt.axis("off")

    except Exception as E:
        print(f"plotWordCloud: {E}")

    return fig

def podcastWordClouds(srcDir, tgtDir=None, stopwords=None):
    """
    plot and save wordclouds of all the transcripted segments in srcDir

    Args:
        srcDir: path to folder storing the .txt files
        tgtDir: path to folder where to store the .png figures
        stopwords: list of str, words to remove from the computation

    Returns:

    """

    _, podcastName = os.path.split(srcDir)

    if tgtDir is None:
        tgtDir = srcDir

    txtFilesList = [x for x in os.listdir(srcDir) if x.endswith(".txt")]

    for txtFile in txtFilesList:
        txtFilePath = os.path.join(srcDir, txtFile)
        txtFileName = txtFile.split(".")[0]

        with open(txtFilePath, "r") as f:
            text = f.read()

        if stopwords:
            STOPWORDS = getStopWords(text, language='it') + stopwords
        else:
            STOPWORDS = getStopWords(text, language='it')

        # if wordcloud summary of whole episode:
        if txtFileName == podcastName:
            extraStopWords = ["radio", "Radio", "Grazie", "grazie", "anni", "bar", "Parigi", "punto", "ciao",
                              "Jack", "Frank", "lockdown", "down", "Roberta", "Francia", "Giuseppe", "ciao ciao",
                              "Gianluca", "Luchino", "Ruggero", "Simone", "fotografo"]
            fig = plotWordCloud(text, maxWords=500, stopwords=set(STOPWORDS + extraStopWords), colormap="cividis")
            figName = f"wordcloud-{txtFileName}.png"

        else:
            fig = plotWordCloud(text, maxWords=300, stopwords=set(STOPWORDS), colormap="cividis")
            figName = f"{txtFileName}.png"

        fig.canvas.start_event_loop(sys.float_info.min)
        plt.savefig(os.path.join(tgtDir, figName), bbox_inches="tight")
        plt.close(fig)



# EXTRA:
def extractSentences(text):
    pass



if __name__ == "__main__":


    PODCASTS = ["lastRadioParty"]
    STOPWORDS = [
        "Luca", "Dani", "Daniele", "Chiara", "Marco", "Leo", "Micol", "Rugge", "Ali", "Pat", "Beppe", "Ste", "Claudio",
        "ragazzi", "ragazze", "raga", "l'altro", "l'altra", "volta", "l'ho", "l'hai", "l'ha", "volte", "l'hanno",
        "dell'altro", "dell'altra", "c'era", "c'erano", "l'ultima", "l'ultimo", "cos'è", "cos'era", "l'abbiamo",
        "cos'erano", "roba", "ni", "realtà", "c'eravate", "c'eravamo", "tanto", "s'è", "glielo", "dall'altro",
        "dall'altra", "cos'altro", "l'è", "cosa", "cose", "tant'è", "meglio", "casi", "po'", "n'era", "gran", "Digli",
        "puntata", "tema", "proposito", "caso", "tipo", "tipa", "tipi", "esempi", "esempio", "Luino", "Ruggiero",
        "anno", "modo", "modi", "Stefano", "Vicente", "Chat", "chat", "com", "posto", "posti", "c'è", "grazie", "radio",
        "Radio", "Grazie", "grazie", "anni", "bar", "Parigi", "punto", "ciao", "Jack", "Frank", "lockdown", "down",
        "Roberta", "Francia", "Giuseppe", "ciao ciao", "Gianluca", "Luchino", "Ruggero", "Simone", "fotografo", "Luca",
        "Dani", "Daniele", "Chiara", "Marco", "Leo", "Micol", "Rugge", "Ali", "Pat", "Beppe", "Ste", "Claudio",
        "ragazzi", "ragazze", "raga", "l'altro", "l'altra", "volta", "l'ho", "l'hai", "l'ha", "volte", "l'hanno",
        "dell'altro", "dell'altra", "c'era", "c'erano", "l'ultima", "l'ultimo", "cos'è", "cos'era", "l'abbiamo",
        "cos'erano", "roba", "ni", "realtà", "c'eravate", "c'eravamo", "tanto", "s'è", "glielo", "dall'altro",
        "dall'altra", "cos'altro", "l'è", "cosa", "cose", "tant'è", "meglio", "casi", "po'", "n'era", "gran", "Digli",
        "puntata", "tema", "proposito", "caso", "tipo", "tipa", "tipi", "esempi", "esempio", "Luino", "Ruggiero",
        "anno", "modo", "modi", "Stefano", "Vicente", "Chat", "chat", "com", "esatto", "Esatto", "un'app", "dov'è",
        "dov'era", "dov'erano", "anch'io", "parte", "domanda", "Federica", "Stefania", "frattempo", "c'ho", "c'hai",
        "c'hanno", "pezzo", "un'altra", "un'altro", "pezzi", "parti", "Marta", "d'accordo", "saluto", "saluti", "sacco",
        "sacchi", "Luigi", "senso", "motivo", "Flora", "notaio", "ore", "ora", "momento", "attimo", "cazzo", "merda",
        "Stefy", "Luca", "Dani", "Daniele", "Chiara", "Marco", "Leo", "Micol", "Rugge", "Ali", "Pat", "Beppe", "Ste",
        "ragazzi", "ragazze", "raga", "l'altro", "l'altra", "volta", "l'ho", "l'hai", "l'ha", "volte", "l'hanno",
        "dell'altro", "dell'altra", "c'era", "c'erano", "l'ultima", "l'ultimo", "cos'è", "cos'era", "l'abbiamo",
        "cos'erano", "roba", "ni", "realtà", "c'eravate", "c'eravamo", "tanto", "s'è", "glielo", "dall'altro",
        "dall'altra", "cos'altro", "l'è", "cosa", "cose", "tant'è", "meglio", "casi", "po'", "n'era", "gran", "Digli",
        "puntata", "tema", "proposito", "caso", "tipo", "tipa", "tipi", "esempi", "esempio", "Luino", "Ruggiero",
        "anno", "modo", "modi", "Stefano", "Vicente", "Chat", "chat", "com", "esatto", "Esatto", "un'app", "dov'è",
        "dov'era", "dov'erano", "anch'io", "l'unico", "l'unica", "Ale", "Francesco", "quest'anno", "un'altro",
        "un'altra", "metà", "paio", "l'anno", "Alice", "giro", "attenzione", "n'è", "dell'anno", "mezzo", "solito",
        "De", "dell'uomo", "musica", "song",
    ]

    # src and tgt dirs:
    srcDir = r"P:\WORK\PYTHONPATH\RUG\projects\autoradiolockdown\ruggero-dev\autolog\audio\puntate"
    mp3Dir = os.path.join(srcDir, "mp3")
    txtDir = os.path.join(srcDir, "txt")

    # segment:
    mp3FileNames = [x + ".mp3" for x in PODCASTS]
    lstFileSrc = [os.path.join(mp3Dir, file) for file in mp3FileNames]
    doSegmentation(lstFileSrc)
    plotAudienceBySegment(lstFileSrc)

    # speech-to-text:
    getTranscriptionFromSegments(podcast=PODCASTS)

    # word cloud:
    for podcast in PODCASTS:
        podcastWordClouds(srcDir=os.path.join(txtDir, podcast), stopwords=STOPWORDS)







