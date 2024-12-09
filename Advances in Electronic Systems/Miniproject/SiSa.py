import numpy as np

class ResourceAllocator:

    def __init__(self, config, allocator_name="random"):

        self.num_of_channels = config.n_subchannel

        self.num_of_subnetworks = config.num_of_subnetworks

        self.allocator_name = allocator_name

        self.max_power = config.max_power

        self.subn_channel_index = np.zeros((config.num_of_subnetworks, 1))

        self.subn_power_value = self.max_power * np.ones((config.num_of_subnetworks, 1))

        self.bandwidth = config.bandwidth

        self.ch_bandwidth = config.ch_bandwidth

        self.noise_power = config.noise_power

        self.capacity = list()

    def allocate(self, ch_gain):

        channel_info_intraIBS = [
            np.diag(ch_gain).reshape((-1, 1)) for i in range(self.num_of_channels)
        ]

        channel_info_interIBS = [
            ch_gain * (1 - np.eye(self.num_of_subnetworks))
            for i in range(self.num_of_channels)
        ]

        if self.allocator_name == "Random":

            self.subn_power_value = np.random.uniform(
                low=0, high=self.max_power, size=(self.num_of_subnetworks, 1)
            )

            ind = np.random.randint(
                0, self.num_of_channels, (self.num_of_subnetworks, 1)
            )

            self.subn_channel_index = ind.astype(np.int64)

            # self.step_time = end - start
            helper_var = self.subn_channel_index.max() + 1
            one_hot = np.zeros(
                (self.subn_channel_index.size, 4)
            )

            one_hot[
                np.arange(self.subn_channel_index.size),
                self.subn_channel_index.squeeze(),
            ] = 1

        # Sequential Iterative Sub-band Allocation

        elif self.allocator_name == "SISA":

            self.subn_power_value = self.max_power * np.ones(
                (self.num_of_subnetworks, 1)
            )

            self.subn_channel_index = np.zeros((self.num_of_subnetworks, 1))

            if self.num_of_subnetworks <= self.num_of_channels:

                self.subn_channel_index = np.linspace(
                    0,
                    self.num_of_subnetworks - 1,
                    num=self.num_of_subnetworks,
                    dtype=np.int32,
                ).reshape((-1, 1))

            else:

                w = np.zeros(
                    (
                        self.num_of_subnetworks,
                        self.num_of_subnetworks,
                        self.num_of_channels,
                    )
                )

                a_t = np.random.randint(
                    self.num_of_channels, size=(self.num_of_subnetworks, 1)
                )

                b_t = []

                for k in range(self.num_of_channels):

                    w[:, :, k] = (np.abs(channel_info_interIBS[k])) / (
                        np.abs(
                            np.tile(
                                channel_info_intraIBS[k].reshape((-1, 1)),
                                (1, self.num_of_subnetworks),
                            )
                        )
                    )

                    b_t.append(np.where(a_t == k)[0])

                w_k = np.zeros((self.num_of_channels, 1))

                for l in range(5):

                    for n in range(self.num_of_subnetworks):

                        for k in range(self.num_of_channels):

                            w_k[k] = np.sum(w[n, b_t[k], k]) + np.sum(w[b_t[k], n, k])

                        a_t[n] = np.argmin(w_k)

                        b_t = []

                        for k in range(self.num_of_channels):

                            b_t.append(np.where(a_t == k)[0])

                ind = a_t

                self.subn_channel_index = ind.astype(np.int64)

                one_hot = np.zeros(
                    (self.subn_channel_index.size, self.subn_channel_index.max() + 1)
                )

                one_hot[
                    np.arange(self.subn_channel_index.size),
                    self.subn_channel_index.squeeze(),
                ] = 1
        return one_hot

