from typing import Callable, List, Union

from splitfleet.server.server_model.server_model import ServerModel
from splitfleet.server.server_model.manager import ServerModelManager
from splitfleet.common import ServerModelFitRes, ServerModelEvaluateIns, ServerModelFitIns


class GrpcServerModelManager(ServerModelManager):

    def __init__(
        self,
        init_server_model_fn: Callable[[], ServerModel],
        *,
        persistent_models: bool = False,
    ) -> None:
        super().__init__()
        self.init_server_model_fn = lambda: init_server_model_fn().to_server_model()
        self.persistent_models = bool(persistent_models)
        self.client_locks = {}
        self.spanned_server_models = {}
        self.active_server_model_ids = []

    def get_server_model(self, sid) -> ServerModel:
        return self.spanned_server_models[sid]

    def collect_server_fit_results(self) -> List[ServerModelFitRes]:
        out = []
        for sid in self.active_server_model_ids:
            sm = self.spanned_server_models[sid]
            res = sm.get_fit_result()
            res.sid = sid
            out.append(res)
        return out

    def end_round(self):
        self.active_server_model_ids = []
        if not self.persistent_models:
            self.spanned_server_models = {}

    def initialize_server_models(
        self,
        configs: List[Union[ServerModelEvaluateIns, ServerModelFitIns]]
    ):
        self.active_server_model_ids = [config.sid for config in configs]
        if len(configs) == 0:
            # no need for any server model
            return

        is_training = isinstance(configs[0], ServerModelFitIns)
        for config in configs:
            sm = self.spanned_server_models.get(config.sid)
            if sm is None or not self.persistent_models:
                sm = self.init_server_model_fn()

            if is_training:
                sm.configure_fit(config)
            else:
                sm.configure_evaluate(config)

            self.spanned_server_models[config.sid] = sm

    def get_server_model_ids(self):
        return list(self.spanned_server_models.keys())
