from anomaly_detection.dpa.pg_networks import ProgGrowNetworks, AbstractNetwork


class FakeNetwork(AbstractNetwork):
    def __init__(self):
        super().__init__(None)

    def get_progress(self):
        return None

    def set_progress(self, progress):
        pass

    def forward(self, x):
        return None


class FakeProgGrowNetworks(ProgGrowNetworks):
    def __init__(self):
        super(FakeProgGrowNetworks, self).__init__()

    def get_net(self, stage, resolution):
        return FakeNetwork()
