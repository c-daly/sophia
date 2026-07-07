from sophia.embodiment.inhabitant import Inhabitant


class FakeEmb:
    def __init__(self):
        self.sent = []
        self._obs = {"odom": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}

    def describe(self):
        class S:  # minimal manifest stand-in
            entity_id = "creature-0"

            def actuator(self, n):
                assert n == "cmd_vel"
                return object()

        return S()

    def read(self):
        class R:
            pass

        r = R()
        r.obs = self._obs
        r.sim_time = 1.0
        return r

    def command(self, cmd):
        self.sent.append(cmd)


def test_perceive_returns_odom():
    inh = Inhabitant(FakeEmb())
    assert inh.perceive()["odom"][3] == 0.0


def test_act_emits_forward_then_turn():
    emb = FakeEmb()
    inh = Inhabitant(emb)
    fwd = inh.act(0)  # scripted step 0 = go forward
    turn = inh.act(10)  # scripted step 10 = yaw
    assert fwd["cmd_vel"][0] > 0 and fwd["cmd_vel"][1] == 0.0
    assert turn["cmd_vel"][1] != 0.0
    assert emb.sent == [fwd, turn]


def test_inhabitant_binds_to_cmd_vel_actuator():
    Inhabitant(FakeEmb())  # asserts the manifest advertises cmd_vel; raises if not
