## Guide to using the source encoding (meta)

If you want to define the encoded source (several shots in one round), you need to define each round seperately.

* define the source wavelet, the shape of one encoded source is `[n_source, n_t]` 
* define the class of `Source`
* using the function of `add_encoded_sources` to add sources at each round.
* the finally shape of encoded source is `[n_round, n_source, n_t]`

```python
# Define source positions in the model
src_z = np.array([1 for i in range(2, nx-1, 5)])  # Z-coordinates for sources
src_x = np.array([i for i in range(2, nx-1, 5)])  # X-coordinates for sources

# Generate wavelet for the source
src_t, src_v = wavelet(nt, dt, f0, amp0=1)  # Create time and wavelet amplitude
src_v = integrate.cumtrapz(src_v, axis=-1, initial=0)  # Integrate wavelet to get velocity

# Compute time delays
ref_x, ref_z = 0, 0  # Reference point for delay calculation
delays = np.sqrt(((src_x - ref_x)*dx)**2 + ((src_z - ref_z)*dz)**2) / 2000  # Time delay for each source

# Apply time delays and sum sources
encoded_source = []  # Initialize encoded source
for i in range(len(src_x)):
    delay_samples = int(delays[i] / dt)  # Convert delay to sample index
    shifted_wavelet = np.zeros(nt)  # Create a zero-padded wavelet
    if delay_samples < nt:
        shifted_wavelet[delay_samples:] = src_v[:nt - delay_samples]  # Apply delay
    encoded_source.append(shifted_wavelet)  # Sum all delayed wavelets

source = Source(nt=nt, dt=dt, f0=f0)  # Initialize source object

source.add_encoded_sources(src_x=np.array([src_x[0::4]]),src_z=np.array([src_z[0::4]]),src_wavelet=np.array([encoded_source[0::4]]),src_type="mt", src_mt=np.array([[1,0,0],[0,1,0],[0,0,1]]))
source.add_encoded_sources(src_x=np.array([src_x[1::4]]),src_z=np.array([src_z[1::4]]),src_wavelet=np.array([encoded_source[1::4]]),src_type="mt", src_mt=np.array([[1,0,0],[0,1,0],[0,0,1]]))
source.add_encoded_sources(src_x=np.array([src_x[2::4]]),src_z=np.array([src_z[2::4]]),src_wavelet=np.array([encoded_source[2::4]]),src_type="mt", src_mt=np.array([[1,0,0],[0,1,0],[0,0,1]]))
source.add_encoded_sources(src_x=np.array([src_x[3::4]]),src_z=np.array([src_z[3::4]]),src_wavelet=np.array([encoded_source[3::4]]),src_type="mt", src_mt=np.array([[1,0,0],[0,1,0],[0,0,1]]))
source.add_encoded_sources(src_x=np.array([src_x[0::4]]),src_z=np.array([src_z[0::4]]),src_wavelet=np.array([encoded_source[0::4][::-1]]),src_type="mt", src_mt=np.array([[1,0,0],[0,1,0],[0,0,1]]))
source.add_encoded_sources(src_x=np.array([src_x[1::4]]),src_z=np.array([src_z[1::4]]),src_wavelet=np.array([encoded_source[1::4][::-1]]),src_type="mt", src_mt=np.array([[1,0,0],[0,1,0],[0,0,1]]))
source.add_encoded_sources(src_x=np.array([src_x[2::4]]),src_z=np.array([src_z[2::4]]),src_wavelet=np.array([encoded_source[2::4][::-1]]),src_type="mt", src_mt=np.array([[1,0,0],[0,1,0],[0,0,1]]))
source.add_encoded_sources(src_x=np.array([src_x[3::4]]),src_z=np.array([src_z[3::4]]),src_wavelet=np.array([encoded_source[3::4][::-1]]),src_type="mt", src_mt=np.array([[1,0,0],[0,1,0],[0,0,1]]))

```

> the `forward_kernel` will recognize the encoded source automatically.