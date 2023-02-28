if __name__ == '__main__':
    import sys
    from pathlib import Path

    project_root = Path(
        __file__).parent.parent.parent.absolute()  # /home/adapting/git/leoxiang66/idp_LiteratureResearch_Tool
    sys.path.append(project_root.__str__())

    import torch
    from lrt.clustering.models.keyBartPlus import *
    from lrt.clustering.models.adapter import *
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import os

    ####################### Adapter Test #############################
    input_dim = 1024
    adapter_hid_dim = 256
    adapter = Adapter(input_dim,adapter_hid_dim)

    data = torch.randn(10, 20, input_dim)

    tmp = adapter(data)

    assert data.size() == tmp.size()
    ####################### Adapter Test #############################

    ####################### BartDecoderPlus Test #############################
    keyBart = AutoModelForSeq2SeqLM.from_pretrained("bloomberg/KeyBART")
    bartDecoderP = BartDecoderPlus(keyBart, 100)
    tmp = bartDecoderP(inputs_embeds=data,
                       output_attentions = True,
                       output_hidden_states = True,
                       encoder_hidden_states = data
                       )
    print(type(tmp))
    # print(tmp.__dict__)
    print(dir(tmp))
    last_hid_states = tmp.last_hidden_state
    hidden_states = tmp.hidden_states
    attentions = tmp.attentions
    cross_attention = tmp.cross_attentions
    print(last_hid_states.shape)
    print(hidden_states.__len__())
    print(attentions.__len__())
    print(len(cross_attention))
    # print(cross_attention[0])
    print(cross_attention[0].shape)

    ####################### BartDecoderPlus Test #############################

    ####################### BartPlus Test #############################
    bartP = BartPlus(keyBart,100)
    tmp = bartP(
        inputs_embeds = data,
        decoder_inputs_embeds = data,
        output_attentions=True,
        output_hidden_states=True,
                )
    print(type(tmp))
    # print(tmp.__dict__)
    print(dir(tmp))
    last_hid_states = tmp.last_hidden_state
    hidden_states = tmp.decoder_hidden_states
    attentions = tmp.decoder_attentions
    cross_attention = tmp.cross_attentions
    print(last_hid_states.shape)
    print(hidden_states.__len__())
    print(attentions.__len__())
    print(len(cross_attention))
    # print(cross_attention[0])
    print(cross_attention[0].shape)
    ####################### BartPlus Test #############################

    ####################### Summary #############################
    from torchinfo import summary

    summary(bartP)
    # summary(bartDecoderP)
    ####################### Summary #############################

    ####################### KeyBartAdapter Test #############################
    keybart_adapter = KeyBartAdapter(100)
    tmp = keybart_adapter(
        inputs_embeds=data,
        decoder_inputs_embeds=data,
        output_attentions=True,
        output_hidden_states=True,
    )
    print(type(tmp))
    # print(tmp.__dict__)
    print(dir(tmp))
    last_hid_states = tmp.encoder_last_hidden_state
    hidden_states = tmp.decoder_hidden_states
    attentions = tmp.decoder_attentions
    cross_attention = tmp.cross_attentions
    print(last_hid_states.shape)
    print(hidden_states.__len__())
    print(attentions.__len__())
    print(len(cross_attention))
    # print(cross_attention[0])
    print(cross_attention[0].shape)
    summary(keybart_adapter)
    ####################### KeyBartAdapter Test #############################