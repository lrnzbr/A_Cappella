import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import write

import wave
from allantools import noise

def speedx(sound_array, factor):
    """ Multiplies the sound's speed by some `factor` """
    indices = np.round( np.arange(0, len(sound_array), factor) )
    indices = indices[indices < len(sound_array)].astype(int)
    return sound_array[ indices.astype(int) ]

def stretch(sound_array, f, window_size, h):
    """ Stretches the sound by a factor `f` """

    phase  = np.zeros(window_size)
    hanning_window = np.hanning(window_size)
    result = np.zeros( len(sound_array) /f + window_size)

    for i in np.arange(0, len(sound_array)-(window_size+h), h*f):

        # two potentially overlapping subarrays
        a1 = sound_array[i: i + window_size]
        a2 = sound_array[i + h: i + window_size + h]

        # resynchronize the second array on the first
        s1 =  np.fft.fft(hanning_window * a1)
        s2 =  np.fft.fft(hanning_window * a2)
        phase = (phase + np.angle(s2/s1)) % 2*np.pi
        a2_rephased = np.fft.ifft(np.abs(s2)*np.exp(1j*phase))

        # add to result
        i2 = int(i/f)
        result[i2 : i2 + window_size] += hanning_window*a2_rephased.astype(result[i2 : i2 + window_size].dtype)

    result = ((2**(16-4)) * result/result.max()) # normalize (16bit)

    return result.astype('int16')

def pitchshift(snd_array, n, window_size=2**13, h=2**11):
    """ Changes the pitch of a sound by ``n`` semitones. """
    factor = 2**(1.0 * n / 12.0)
    snd_array1 = snd_array[:,0]
    snd_array2 = snd_array[:,1]

    stretched1 = stretch(snd_array1, 1.0/factor, window_size, h)
    stretched2 = stretch(snd_array2, 1.0/factor, window_size, h)
    stretched = np.array([stretched1, stretched2]).T
    filename = "pitch_test.wav"
    write(filename, 44100, speedx(stretched[window_size:], factor))

    return


def add_noise(snd_array,fps):
    noise1 = noise.white(len(snd_array))
    noise1 = map(lambda x: int(x/.01), noise1)

    noise2 = np.array([noise1,noise1]).T
    noisy_array = noise2 + snd_array

    filename = "noisy_test.wav"

    scaled = np.int16(noisy_array/np.max(np.abs(noisy_array)) * 32767)
    write(filename, 44100, scaled)
    return



if __name__ == "__main__":

    fps, bowl_sound = wavfile.read("bowl.wav")
    tones = range(-25,25)
    transposed = [pitchshift(bowl_sound, n) for n in tones]
