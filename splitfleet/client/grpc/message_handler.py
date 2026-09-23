from flwr.client.client import (
    maybe_call_evaluate,
    maybe_call_fit,
    maybe_call_get_parameters,
    maybe_call_get_properties,
)
from typing import Callable

from flwr.common import Message
from flwr.app.message_type import MessageType
from flwr.common.constant import MessageTypeLegacy
from flwr.common.recorddict_compat import (
    evaluateres_to_recorddict,
    fitres_to_recorddict,
    getparametersres_to_recorddict,
    getpropertiesres_to_recorddict,
    recorddict_to_evaluateins,
    recorddict_to_fitins,
    recorddict_to_getparametersins,
    recorddict_to_getpropertiesins,
)

from splitfleet.common.constants import CLIENT_ID_CONFIG_KEY
from splitfleet.client.client import Client
from splitfleet.proto.server_model_pb2_grpc import ServerModelStub
from splitfleet.server.server_model.proxy.grpc_server_model_proxy import GrpcServerModelProxy


def handle_message(
    client_fn: Callable[[str], Client], message: Message, server_model_stub: ServerModelStub
) -> Message:
    """Dispatch a Flower instruction with its companion server-model proxy."""
    cid = str(message.metadata.src_node_id)
    assert len(message.content.configs_records) == 1
    config_record = next(iter(message.content.configs_records.values()))

    server_cid = ""
    if CLIENT_ID_CONFIG_KEY in config_record:
        server_cid = config_record.pop(CLIENT_ID_CONFIG_KEY)

    client = client_fn(cid).to_client()

    server_model_proxy = GrpcServerModelProxy(stub=server_model_stub, cid=server_cid)
    client.set_server_model_proxy(
        server_model_proxy
    )
    message_type = message.metadata.message_type

    # Handle GetPropertiesIns
    if message_type == MessageTypeLegacy.GET_PROPERTIES:
        get_properties_res = maybe_call_get_properties(
            client=client,
            get_properties_ins=recorddict_to_getpropertiesins(message.content),
        )
        out_recordset = getpropertiesres_to_recorddict(get_properties_res)
    # Handle GetParametersIns
    elif message_type == MessageTypeLegacy.GET_PARAMETERS:
        get_parameters_res = maybe_call_get_parameters(
            client=client,
            get_parameters_ins=recorddict_to_getparametersins(message.content),
        )
        out_recordset = getparametersres_to_recorddict(
            get_parameters_res, keep_input=False
        )
    # Handle FitIns
    elif message_type == MessageType.TRAIN:
        fit_res = maybe_call_fit(
            client=client,
            fit_ins=recorddict_to_fitins(message.content, keep_input=True),
        )
        out_recordset = fitres_to_recorddict(fit_res, keep_input=False)
    # Handle EvaluateIns
    elif message_type == MessageType.EVALUATE:
        evaluate_res = maybe_call_evaluate(
            client=client,
            evaluate_ins=recorddict_to_evaluateins(message.content, keep_input=True),
        )
        out_recordset = evaluateres_to_recorddict(evaluate_res)
    else:
        raise ValueError(f"Invalid message type: {message_type}")

    server_model_proxy.close_stream()

    # Return Message
    return message.create_reply(out_recordset)
