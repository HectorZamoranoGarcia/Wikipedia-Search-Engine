"""
Autores:
- Héctor Zamorano García

Funcionalidades Implementadas:
- Búsqueda y evaluación booleana (AND, NOT) usando algoritmo de cruce por punteros.
- Búsquedas posicionales (uso de comillas).
- Búsqueda semántica (Sentence-BERT, KDTree).
- Reranking semántico.
"""

# versión 1.2

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, List, Union, Dict
import pickle
try:
    import nltk
    from SAR_semantics import SentenceBertEmbeddingModel, BetoEmbeddingCLSModel, BetoEmbeddingModel, SpacyStaticModel
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False


## UTILIZAR PARA LA AMPLIACION
# Selecciona un modelo semántico
SEMANTIC_MODEL = "SBERT"
#SEMANTIC_MODEL = "BetoCLS"
#SEMANTIC_MODEL = "Beto"
#SEMANTIC_MODEL = "Spacy"
#SEMANTIC_MODEL = "Spacy_noSW_noA"

def create_semantic_model(modelname):
    assert modelname in ("SBERT", "BetoCLS", "Beto", "Spacy", "Spacy_noSW_noA")
    
    if modelname == "SBERT": return SentenceBertEmbeddingModel()    
    elif modelname == "BetoCLS": return BetoEmbeddingCLSModel()
    elif modelname == "Beto": return BetoEmbeddingModel()
    elif modelname == "Spacy": SpacyStaticModel(remove_stopwords=False, remove_noalpha=False)
    return SpacyStaticModel()


class SAR_Indexer:
    """
    Prototipo de la clase para realizar la indexacion y la recuperacion de artículos de Wikipedia
        
        Preparada para todas las ampliaciones:
          posicionales + busqueda semántica + ranking semántico

    Se deben completar los metodos que se indica.
    Se pueden añadir nuevas variables y nuevos metodos
    Los metodos que se añadan se deberan documentar en el codigo y explicar en la memoria
    """

    # campo que se indexa
    DEFAULT_FIELD = 'all'
    # numero maximo de documento a mostrar cuando self.show_all es False
    SHOW_MAX = 10


    all_atribs = ['urls', 'index', 'docs', 'articles', 'tokenizer', 'show_all',
                  'positional',
                  "semantic", "chuncks", "embeddings", "chunck_index", "kdtree", "artid_to_emb"]


    def __init__(self):
        """
        Constructor de la clase SAR_Indexer.
        NECESARIO PARA LA VERSION MINIMA

        Incluye todas las variables necesaria pero
        	puedes añadir más variables si las necesitas. 

        """
        self.urls = set() # hash para las urls procesadas,
        self.index = {} # hash para el indice invertido de terminos --> clave: termino, valor: posting list
        self.docs = {} # diccionario de terminos --> clave: entero(docid),  valor: ruta del fichero.
        self.articles = {} # hash de articulos --> clave entero (artid), valor: la info necesaria para diferencia los artículos dentro de su fichero
        self.tokenizer = re.compile(r"\W+") # expresion regular para hacer la tokenizacion
        self.show_all = False # valor por defecto, se cambia con self.set_showall()
        self.positional = False # indica si el indice es posicional

        # PARA LA AMPLIACION
        self.semantic = None
        self.chuncks = []
        self.embeddings = []
        self.chunck_index = []
        self.artid_to_emb = {}
        self.kdtree = None
        self.semantic_threshold = None
        self.semantic_ranking = None # ¿¿ ranking de consultas binarias ??
        self.model = None
        self.MAX_EMBEDDINGS = 200 # número máximo de embedding que se extraen del kdtree en una consulta
        
        


    ###############################
    ###                         ###
    ###      CONFIGURACION      ###
    ###                         ###
    ###############################


    def set_showall(self, v:bool):
        """

        Cambia el modo de mostrar los resultados.

        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_all es True se mostraran todos los resultados el lugar de un maximo de self.SHOW_MAX, no aplicable a la opcion -C

        """
        self.show_all = v


    def set_semantic_threshold(self, v:float):
        """

        Cambia el umbral para la búsqueda semántica.

        input: "v" booleano.

        UTIL PARA LA AMPLIACIÓN

        si self.semantic es False el umbral no tendrá efecto.

        """
        self.semantic_threshold = v

    def set_semantic_ranking(self, v:bool):
        """

        Cambia el valor de semantic_ranking.

        input: "v" booleano.

        UTIL PARA LA AMPLIACIÓN

        si self.semantic_ranking es True se hará una consulta binaria y los resultados se rankearán por similitud semántica.

        """
        self.semantic_ranking = v


    #############################################
    ###                                       ###
    ###      CARGA Y GUARDADO DEL INDICE      ###
    ###                                       ###
    #############################################


    def save_info(self, filename:str):
        """
        Guarda la información del índice en un fichero en formato binario

        """
        info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'wb') as fh:
            pickle.dump(info, fh)

    def load_info(self, filename:str):
        """
        Carga la información del índice desde un fichero en formato binario

        """
        #info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'rb') as fh:
            info = pickle.load(fh)
        atrs = info[0]
        for name, val in zip(atrs, info[1:]):
            setattr(self, name, val)


    ###############################
    ###                         ###
    ###   SIMILITUD SEMANTICA   ###
    ###                         ###
    ###############################

            
    def load_semantic_model(self, modelname:str=SEMANTIC_MODEL):
        """
    
        Carga el modelo de embeddings para la búsqueda semántica.
        Solo se debe cargar una vez
        
        """
        if self.model is None:
            print(f"loading {modelname} model ... ",end="", file=sys.stderr)             
            self.model = create_semantic_model(modelname)
            print("done!", file=sys.stderr)

            

    def update_chuncks(self, txt:str, artid:int):
        """
        
        Añade los chuncks (frases en nuestro caso) del texto "txt" correspondiente al articulo "artid" en la lista de chuncks
        Pasos:
            1 - extraer los chuncks de txt, en nuestro caso son las frases. Se debe utilizar "sent_tokenize" de la librería "nltk"
            2 - actualizar los atributos que consideres necesarios: self.chuncks, self.embeddings, self.chunck_index y self.artid_to_emb.
        
        """
        # 1 - Extraer frases del texto usando sent_tokenize
        sentences = nltk.sent_tokenize(txt, language='spanish')
        
        # 2 - Actualizar estructuras
        start_idx = len(self.chuncks)
        self.chuncks.extend(sentences)
        
        # Mapear cada chunk a su artículo
        for i in range(len(sentences)):
            self.chunck_index.append(artid)
        
        # Mapear artículo a sus índices de chunks
        self.artid_to_emb[artid] = list(range(start_idx, start_idx + len(sentences)))


    def create_kdtree(self):
        """
        
        Crea el kdtree utilizando un objeto de la librería SAR_semantics
        Solo se debe crear una vez despues de indexar todos los documentos
        
        # 1: Se debe llamar al método fit del modelo semántico
        # 2: Opcionalmente se puede guardar información del modelo semántico (kdtree y/o embeddings) en el SAR_Indexer
        
        """
        print(f"Creating kdtree ...", end="")
        # 1 - Crear embeddings y kdtree con fit
        self.model.fit(self.chuncks)
        # 2 - Guardar kdtree y embeddings en el indexer
        self.kdtree = self.model.kdtree
        self.embeddings = self.model.embeddings
        print("done!")


        
    def solve_semantic_query(self, query:str):
        """

        Resuelve una consulta utilizando el modelo semántico.
        Pasos:
            1 - utiliza el método query del modelo sémantico
            2 - devuelve top_k resultados, inicialmente top_k puede ser MAX_EMBEDDINGS
            3 - si el último resultado tiene una distancia <= self.semantic_threshold 
                  ==> no se han recuperado todos los resultado: vuelve a 2 aumentando top_k
            4 - también se puede salir si recuperamos todos los embeddings
            5 - tenemos una lista de chuncks que se debe pasar a artículos
        """

        self.load_semantic_model()
        # Restaurar kdtree y embeddings en el modelo
        if self.model.kdtree is None and self.kdtree is not None:
            self.model.set_kdtree(self.kdtree)
            self.model.set_embeddings(self.embeddings)
        
        total = len(self.chuncks)
        if total == 0:
            return []

        top_k = min(self.MAX_EMBEDDINGS, total)
        
        while True:
            results = self.model.query(query, top_k=top_k)
            if len(results) == 0:
                return []
            last_dist = results[-1][0]
            if last_dist > self.semantic_threshold or top_k >= total:
                break
            top_k = min(top_k * 2, total)

        # Filtrar por umbral y mapear a artículos sin repetidos
        seen = set()
        article_list = []
        for dist, idx in results:
            if dist > self.semantic_threshold:
                break
            artid = self.chunck_index[idx]
            if artid not in seen:
                seen.add(artid)
                article_list.append(artid)
        
        return article_list


    def semantic_reranking(self, query:str, articles: List[int]):
        """

        Ordena los articulos en la lista 'article' por similitud a la consulta 'query'.
        Pasos:
            1 - utiliza el método query del modelo sémantico
            2 - devuelve top_k resultado, inicialmente top_k puede ser MAX_EMBEDDINGS
            3 - a partir de los chuncks se deben obtener los artículos
            3 - si entre los artículos recuperados NO estan todos los obtenidos por la RI binaria
                  ==> no se han recuperado todos los resultado: vuelve a 2 aumentando top_k
            4 - se utiliza la lista ordenada del kdtree para ordenar la lista "articles"
        """
        
        self.load_semantic_model()
        # Restaurar kdtree y embeddings en el modelo
        if self.model.kdtree is None and self.kdtree is not None:
            self.model.set_kdtree(self.kdtree)
            self.model.set_embeddings(self.embeddings)

        articles_set = set(articles)
        total = len(self.chuncks)
        if total == 0:
            return articles

        top_k = min(self.MAX_EMBEDDINGS, total)
        
        while True:
            results = self.model.query(query, top_k=top_k)
            found = set()
            for dist, idx in results:
                artid = self.chunck_index[idx]
                if artid in articles_set:
                    found.add(artid)
            if found >= articles_set or top_k >= total:
                break
            top_k = min(top_k * 2, total)
        
        # Ordenar según la similitud semántica
        ordered = []
        seen = set()
        for dist, idx in results:
            artid = self.chunck_index[idx]
            if artid in articles_set and artid not in seen:
                seen.add(artid)
                ordered.append(artid)
        
        # Añadir artículos que no aparecieron en los resultados semánticos
        for artid in articles:
            if artid not in seen:
                ordered.append(artid)
        
        return ordered
    

    ###############################
    ###                         ###
    ###   PARTE 1: INDEXACION   ###
    ###                         ###
    ###############################

    def already_in_index(self, article:Dict) -> bool:
        """

        Args:
            article (Dict): diccionario con la información de un artículo

        Returns:
            bool: True si el artículo ya está indexado, False en caso contrario
        """
        return article['url'] in self.urls


    def index_dir(self, root:str, **args):
        """

        Recorre recursivamente el directorio o fichero "root"
        NECESARIO PARA TODAS LAS VERSIONES

        Recorre recursivamente el directorio "root"  y indexa su contenido
        los argumentos adicionales "**args" solo son necesarios para las funcionalidades ampliadas

        """
        self.positional = args['positional']
        self.semantic = args['semantic']
        if self.semantic is True:
            self.load_semantic_model()


        file_or_dir = Path(root)

        if file_or_dir.is_file():
            # is a file
            self.index_file(root)
        elif file_or_dir.is_dir():
            # is a directory
            for d, _, files in os.walk(root):
                for filename in sorted(files):
                    if filename.endswith('.json'):
                        fullname = os.path.join(d, filename)
                        self.index_file(fullname)
        else:
            print(f"ERROR:{root} is not a file nor directory!", file=sys.stderr)
            sys.exit(-1)

        ######################################################
        ## COMPLETAR SI ES NECESARIO FUNCIONALIDADES EXTRA  ##
        ######################################################
        # Crear kdtree si se ha activado la búsqueda semántica
        if self.semantic is True:
            self.create_kdtree()

        
    def parse_article(self, raw_line:str) -> Dict[str, str]:
        """
        Crea un diccionario a partir de una linea que representa un artículo del crawler

        Args:
            raw_line: una linea del fichero generado por el crawler

        Returns:
            Dict[str, str]: claves: 'url', 'title', 'summary', 'all', 'section-name'
        """
        
        article = json.loads(raw_line)
        sec_names = []
        txt_secs = ''
        for sec in article['sections']:
            txt_secs += sec['name'] + '\n' + sec['text'] + '\n'
            txt_secs += '\n'.join(subsec['name'] + '\n' + subsec['text'] + '\n' for subsec in sec['subsections']) + '\n\n'
            sec_names.append(sec['name'])
            sec_names.extend(subsec['name'] for subsec in sec['subsections'])
        article.pop('sections') # no la necesitamos
        article['all'] = article['title'] + '\n\n' + article['summary'] + '\n\n' + txt_secs
        article['section-name'] = '\n'.join(sec_names)

        return article


    def index_file(self, filename:str):
        """

        Indexa el contenido de un fichero.

        input: "filename" es el nombre de un fichero generado por el Crawler cada línea es un objeto json
            con la información de un artículo de la Wikipedia

        NECESARIO PARA TODAS LAS VERSIONES

        dependiendo del valor de self.positional se debe ampliar el indexado

        """
        # Asignar docid al fichero
        docid = len(self.docs)
        self.docs[docid] = filename

        for i, line in enumerate(open(filename, encoding='utf-8')):
            j = self.parse_article(line)

            # Comprobar si el artículo ya ha sido indexado (URL duplicada)
            if self.already_in_index(j):
                continue

            # Asignar artid único
            artid = len(self.articles)
            self.urls.add(j['url'])
            # Guardar información del artículo: docid, posición en fichero, url, título
            self.articles[artid] = (docid, i, j['url'], j['title'])

            # Obtener texto del campo a indexar y tokenizar
            text = j[self.DEFAULT_FIELD]
            tokens = self.tokenize(text)

            if self.positional:
                # Índice posicional: cada término almacena lista de (artid, [posiciones])
                for pos, token in enumerate(tokens):
                    if token not in self.index:
                        self.index[token] = []
                    posting = self.index[token]
                    if len(posting) == 0 or posting[-1][0] != artid:
                        posting.append((artid, [pos]))
                    else:
                        posting[-1][1].append(pos)
            else:
                # Índice no posicional: cada término almacena lista de artids
                seen_tokens = set()
                for token in tokens:
                    if token not in seen_tokens:
                        seen_tokens.add(token)
                        if token not in self.index:
                            self.index[token] = []
                        self.index[token].append(artid)

            # Actualizar chunks semánticos si procede
            if self.semantic:
                self.update_chuncks(j[self.DEFAULT_FIELD], artid)


    def tokenize(self, text:str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Tokeniza la cadena "texto" eliminando simbolos no alfanumericos y dividientola por espacios.
        Puedes utilizar la expresion regular 'self.tokenizer'.

        params: 'text': texto a tokenizar

        return: lista de tokens

        """
        return self.tokenizer.sub(' ', text.lower()).split()




    def show_stats(self):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Muestra estadisticas de los indices

        """
        print()
        print("========================================")
        print(f"Number of indexed files: {len(self.docs)}")
        print("----------------------------------------")
        print(f"Number of indexed articles: {len(self.articles)}")
        print("----------------------------------------")
        print("TOKENS:")
        print(f"\t# of tokens in '{self.DEFAULT_FIELD}': {len(self.index)}")
        print("----------------------------------------")
        if self.positional:
            print("Positional queries are allowed.")
        else:
            print("Positional queries are NOT allowed.")
        print("========================================")



    #################################
    ###                           ###
    ###   PARTE 2: RECUPERACION   ###
    ###                           ###
    #################################

    ###################################
    ###                             ###
    ###   PARTE 2.1: RECUPERACION   ###
    ###                             ###
    ###################################


    def solve_query(self, query:str, prev:Dict={}):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una query.
        Debe realizar el parsing de consulta que sera mas o menos complicado en funcion de la ampliacion que se implementen


        param:  "query": cadena con la query
                "prev": incluido por si se quiere hacer una version recursiva. No es necesario utilizarlo.


        return: posting list con el resultado de la query

        """
        
        if query is None or len(query) == 0:
            return [], None

        # Búsqueda semántica pura (con umbral)
        if self.semantic_threshold is not None and self.semantic is True:
            result = self.solve_semantic_query(query)
            return result, None

        # Parsear la query: manejar comillas, NOT e AND implícito
        tokens = []
        i = 0
        q = query.strip()
        while i < len(q):
            if q[i] == '"':
                # Buscar cierre de comillas
                end = q.find('"', i + 1)
                if end == -1:
                    end = len(q)
                tokens.append(q[i:end+1])
                i = end + 1
            elif q[i] == ' ':
                i += 1
            else:
                end = i
                while end < len(q) and q[end] != ' ' and q[end] != '"':
                    end += 1
                tokens.append(q[i:end])
                i = end

        result = None
        negate = False

        for token in tokens:
            if token.upper() == 'NOT':
                negate = True
                continue

            if token.startswith('"') and token.endswith('"'):
                # Consulta posicional
                phrase = token[1:-1]
                terms = self.tokenize(phrase)
                if len(terms) == 0:
                    posting = []
                else:
                    posting = self.get_positionals(terms)
            else:
                posting = self.get_posting(token)

            if negate:
                posting = self.reverse_posting(posting)
                negate = False

            if result is None:
                result = posting
            else:
                result = self.and_posting(result, posting)

        if result is None:
            result = []

        # Reranking semántico si está activado
        if self.semantic_ranking and self.semantic is True and len(result) > 0:
            result = self.semantic_reranking(query, result)

        return result, None




    def get_posting(self, term:str):
        """

        Devuelve la posting list asociada a un termino.
        Puede llamar self.get_positionals: para las búsquedas posicionales.


        param:  "term": termino del que se debe recuperar la posting list.

        return: posting list

        NECESARIO PARA TODAS LAS VERSIONES

        """
        term = term.lower()
        if term not in self.index:
            return []
        if self.positional:
            # Extraer solo los artids de la posting list posicional
            return [artid for artid, _ in self.index[term]]
        else:
            return list(self.index[term])



    def get_positionals(self, terms:str):
        """

        Devuelve la posting list asociada a una secuencia de terminos consecutivos.
        NECESARIO PARA LAS BÚSQUESAS POSICIONALES

        param:  "terms": lista con los terminos consecutivos para recuperar la posting list.

        return: posting list

        """
        if not self.positional or len(terms) == 0:
            return []

        terms = [t.lower() for t in terms]

        for t in terms:
            if t not in self.index:
                return []

        if len(terms) == 1:
            return [artid for artid, _ in self.index[terms[0]]]

        # Intersección posicional pairwise usando punteros (algoritmo teórico)
        current_p = self.index[terms[0]]

        for k in range(1, len(terms)):
            next_p = self.index[terms[k]]
            merged_p = []
            
            i, j = 0, 0
            while i < len(current_p) and j < len(next_p):
                artid_1, pos_1 = current_p[i]
                artid_2, pos_2 = next_p[j]
                
                if artid_1 == artid_2:
                    # Mismo artículo, intersectar posiciones con punteros
                    pos_matches = []
                    pi, pj = 0, 0
                    while pi < len(pos_1) and pj < len(pos_2):
                        diff = pos_2[pj] - pos_1[pi]
                        if diff == 1:
                            pos_matches.append(pos_2[pj])
                            pi += 1
                            pj += 1
                        elif diff > 1:
                            # pos_2 está muy adelante, avanzar pos_1
                            pi += 1
                        else:
                            # pos_2 está por detrás o en la misma posición, avanzar pos_2
                            pj += 1
                            
                    if len(pos_matches) > 0:
                        merged_p.append((artid_1, pos_matches))
                    i += 1
                    j += 1
                elif artid_1 < artid_2:
                    i += 1
                else:
                    j += 1
                    
            current_p = merged_p
            if not current_p:
                return []

        return [artid for artid, _ in current_p]



    def reverse_posting(self, p:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve una posting list con todas las noticias excepto las contenidas en p.
        Util para resolver las queries con NOT.


        param:  "p": posting list


        return: posting list con todos los artid exceptos los contenidos en p

        """
        # Algoritmo de merge para obtener el complemento (sin usar sets)
        all_artids = sorted(self.articles.keys())
        result = []
        i, j = 0, 0
        while i < len(all_artids) and j < len(p):
            if all_artids[i] < p[j]:
                result.append(all_artids[i])
                i += 1
            elif all_artids[i] == p[j]:
                i += 1
                j += 1
            else:
                j += 1
        while i < len(all_artids):
            result.append(all_artids[i])
            i += 1
        return result



    def and_posting(self, p1:list, p2:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el AND de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos en p1 y p2

        """
        # Algoritmo de merge para intersección (sin usar sets)
        result = []
        i, j = 0, 0
        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                result.append(p1[i])
                i += 1
                j += 1
            elif p1[i] < p2[j]:
                i += 1
            else:
                j += 1
        return result




    def minus_posting(self, p1, p2):
        """
        OPCIONAL PARA TODAS LAS VERSIONES

        Calcula el except de dos posting list de forma EFICIENTE.
        Esta funcion se incluye por si es util, no es necesario utilizarla.

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos de p1 y no en p2

        """
        # Algoritmo de merge para diferencia (sin usar sets)
        result = []
        i, j = 0, 0
        while i < len(p1) and j < len(p2):
            if p1[i] < p2[j]:
                result.append(p1[i])
                i += 1
            elif p1[i] == p2[j]:
                i += 1
                j += 1
            else:
                j += 1
        while i < len(p1):
            result.append(p1[i])
            i += 1
        return result




    #####################################
    ###                               ###
    ### PARTE 2.2: MOSTRAR RESULTADOS ###
    ###                               ###
    #####################################

    def solve_and_count(self, ql:List[str], verbose:bool=True) -> List:
        results = []
        for query in ql:
            if len(query) > 0 and query[0] != '#':
                r, _ = self.solve_query(query)
                results.append(len(r))
                if verbose:
                    print(f'{query}\t{len(r)}')
            else:
                results.append(0)
                if verbose:
                    print(query)
        return results


    def solve_and_test(self, ql:List[str]) -> bool:
        errors = False
        for line in ql:
            if len(line) > 0 and line[0] != '#':
                query, ref = line.split('\t')
                reference = int(ref)
                result, _ = self.solve_query(query)
                result = len(result)
                if reference == result:
                    print(f'{query}\t{result}')
                else:
                    print(f'>>>>{query}\t{reference} != {result}<<<<')
                    errors = True
            else:
                print(line)

        return not errors


    def solve_and_show(self, query:str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta y la muestra junto al numero de resultados

        param:  "query": query que se debe resolver.

        return: el numero de artículo recuperadas, para la opcion -T

        """
        result, _ = self.solve_query(query)
        n = len(result)
        print(f"Query: '{query}'")
        print(f"Number of results: {n}")

        # Limitar resultados si no se muestra todo
        show = result if self.show_all else result[:self.SHOW_MAX]

        for i, artid in enumerate(show):
            _, _, url, title = self.articles[artid]
            print(f" {i+1}\t({artid})\t{title}\t{url}")

        return n
