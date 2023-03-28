from typing import List
from sentence_transformers import SentenceTransformer
import torch
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering,SpectralClustering
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,Text2TextGenerationPipeline
from trendflow.inference_hf import InferenceHF
from .dimension_reduction import PCA
from unsupervised_learning.clustering import GaussianMixture
from models import KeyBartAdapter

class Template:
    def __init__(self):
        self.PLM = {
            'sentence-transformer-mini': '''sentence-transformers/all-MiniLM-L6-v2''',
            'sentence-t5-xxl': '''sentence-transformers/sentence-t5-xxl''',
            'all-mpnet-base-v2':'''sentence-transformers/all-mpnet-base-v2'''
        }
        self.dimension_reduction = {
            'pca': PCA,
            'vae': None,
            'cnn': None
        }

        self.clustering = {
            'kmeans-euclidean': KMeans,
            'gmm': GaussianMixture,
            'dbscan':None,
            'agglomerative': None,
            'spectral':None
        }

        self.keywords_extraction = {
            'keyphrase-transformer': '''snrspeaks/KeyPhraseTransformer''',
            'KeyBartAdapter': '''Adapting/KeyBartAdapter''',
            'KeyBart': '''bloomberg/KeyBART'''
        }

template = Template()

def __create_model__(model_ckpt):
    '''

    :param model_ckpt: keys in Template class
    :return: model/function: callable
    '''
    if model_ckpt == '''sentence-transformer-mini''':
        return SentenceTransformer(template.PLM[model_ckpt])
    elif model_ckpt == '''sentence-t5-xxl''':
        return SentenceTransformer(template.PLM[model_ckpt])
    elif model_ckpt == '''all-mpnet-base-v2''':
        return SentenceTransformer(template.PLM[model_ckpt])
    elif model_ckpt == 'none':
        return None
    elif model_ckpt == 'kmeans-cosine':
        def ret(x,k):
            tmp = template.clustering[model_ckpt](
            X=torch.from_numpy(x), num_clusters=k, distance='cosine',
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
            return tmp[0].cpu().detach().numpy(), tmp[1].cpu().detach().numpy()
        return ret
    elif model_ckpt == 'pca':
        pca = template.dimension_reduction[model_ckpt](0.95)
        return pca

    elif model_ckpt =='kmeans-euclidean':
        def ret(x,k):
            tmp = KMeans(n_clusters=k,random_state=50).fit(x)
            return tmp.labels_, tmp.cluster_centers_
        return ret
    elif model_ckpt == 'gmm':
        def ret(x,k):
            model = GaussianMixture(k,50)
            model.fit(x)
            return model.getLabels(), model.getClusterCenters()
        return ret
    elif model_ckpt == 'dbscan':
        def ret(x,k):
            model = DBSCAN(eps=5, min_samples=2)
            return model.fit_predict(x), None
        return ret

    elif model_ckpt == 'agglomerative':
        def  ret(x,k):
            model = AgglomerativeClustering(n_clusters=k)
            return model.fit_predict(x), None
        return ret

    elif model_ckpt == 'spectral':
        def ret(x,k):
            model = SpectralClustering(n_clusters=k, eigen_solver='arpack', assign_labels='kmeans', random_state=50)
            return model.fit_predict(x),None
        return ret

    elif model_ckpt == 'keyphrase-transformer':
        model_ckpt = template.keywords_extraction[model_ckpt]

        def ret(texts: List[str]):
            # first try inference API
            response = InferenceHF.inference(
                inputs=texts,
                model_name=model_ckpt
            )

            # inference failed:
            if not isinstance(response, list):
                tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)
                pipe = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer)

                tmp = pipe(texts)
                results = [
                    set(
                        map(str.strip,
                            x['generated_text'].split('|')  # [str...]
                            )
                    )
                    for x in tmp]  # [{str...}...]

                return results

            # inference sucsess
            else:
                results = [
                    set(
                        map(str.strip,
                            x['generated_text'].split('|')  # [str...]
                            )
                    )
                    for x in response]  # [{str...}...]

                return results

        return ret

    elif model_ckpt == 'KeyBart':
        model_ckpt = template.keywords_extraction[model_ckpt]
        def ret(texts: List[str]):
            # first try inference API
            response = InferenceHF.inference(
                inputs=texts,
                model_name=model_ckpt
            )

            # inference failed:
            if not isinstance(response,list):
                tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)
                pipe = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer)


                tmp = pipe(texts)
                results = [
                    set(
                        map(str.strip,
                            x['generated_text'].split(';')  # [str...]
                            )
                    )
                    for x in tmp]  # [{str...}...]

                return results

            # inference sucsess
            else:
                results = [
                    set(
                        map(str.strip,
                            x['generated_text'].split(';')  # [str...]
                            )
                    )
                    for x in response]  # [{str...}...]

                return results

        return ret

    elif model_ckpt == 'KeyBartAdapter':
        def ret(texts: List[str]):
            model = KeyBartAdapter.from_pretrained('Adapting/KeyBartAdapter',revision='3aee5ecf1703b9955ab0cd1b23208cc54eb17fce', adapter_hid_dim=32)
            tokenizer = AutoTokenizer.from_pretrained("bloomberg/KeyBART")
            pipe = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer)

            tmp = pipe(texts)
            results = [
                set(
                    map(str.strip,
                        x['generated_text'].split(';')  # [str...]
                        )
                )
                for x in tmp]  # [{str...}...]

            return results
        return ret


    else:
        raise RuntimeError(f'The model {model_ckpt} is not supported. Please open an issue on the GitHub about the model.')

