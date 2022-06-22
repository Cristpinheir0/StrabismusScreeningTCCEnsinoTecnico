import streamlit as st
import numpy as np
import cv2
import dlib
from math import sqrt
from PIL import Image

@st.cache
def load_image(img):
    im = Image.open(img)
    return im


def main():
    st.markdown(
        "<h1 style='text-align: center; color: white;'>STRABISMUS SCREENING: Sistema para triagem de estrabismo infantil</h1>",
        unsafe_allow_html=True)
    st.markdown('***')
    st.markdown(
        'Este site realiza a triagem de estrabismo, baseando-se no Teste de Hirschberg. '
        'O teste consiste em posicionar um feixe de luz nos olhos da pessoa '
        'e verificar a posição do reflexo em relação ao centro da pupila. Quando eles não estão no mesmo ponto, '
        'ou seja, estão distantes um do outro, é identificado o desvio ocular. Portanto, a partir deste teste é '
        'possível identificar o estrabismo.')
    st.markdown('***')
    st.markdown(
        '***Para utilizar a ferramenta de triagem é necessário ter uma foto realizando o teste de Hirschberg.***')
    st.markdown('***Orientações para a captura da foto:***')
    st.markdown(
        '- A pessoa que irá realizar o teste deve ficar em um ambiente escuro e a uma distância de aproximadamente 1 metro da câmera;')
    st.markdown('- É necessário ligar o flash;')
    st.markdown('- E no momento da captura a pessoa deve olhar diretamente para o ponto de luz.')
    st.markdown(
        '***Importante:*** As fotos não serão armazenadas pelo site, ou seja, nenhuma informação será mantida ou usada.')
    st.markdown('***')
    st.subheader('Upload da foto:')
    st.markdown('***Clicar em "Browse files" e selecionar a foto.***')
    image_file = st.file_uploader('', type=['jpg', 'png', 'jpeg'])

    if image_file is not None:
        imagem = Image.open(image_file)
        st.markdown('***Imagem selecionada:***')
        st.image(imagem)
        st.markdown('***')
        imagem = np.array(imagem.convert('RGB'))  # CONVERSÃO DA IMAGEM PARA ARRAY

        imagem_cinza = preProcessamento(imagem)  # FUNÇÃO DE PRE PROCESSAMENTO DA IMAGEM:

        rosto, pontos = identificacaoDlib(imagem_cinza)  # FUNÇÃO DE IDENTIFICAÇÃO DOS OLHOS

        pontos_dlib = imagem.copy()  # CÓPIA DA IMAGEM PARA DEMONSTRAR IDENTIFICAÇÕES DOS PONTOS

        # PONTOS DO LANDMARK REFERENTE AOS OLHOS
        olho_direito_pontos, olho_esquerdo_pontos = pontoOlhos(pontos_dlib, pontos)

        # FUNÇÃO DE SEPARAÇÃO DOS OLHOS
        olho_direito, olho_direito_cinza, olho_esquerdo, olho_esquerdo_cinza = separacaoOlhos(imagem,
                                                                                              olho_direito_pontos,
                                                                                              olho_esquerdo_pontos)
        # FUNÇÃO DE APLICAÇÃO DO THRESHOLD NOS OLHOS
        olho_direito_pb, olho_esquerdo_pb = thresholding(olho_direito_cinza, olho_esquerdo_cinza)

        # FUNÇÃO QUE IDENTIFICA O CENTRO DO OLHO
        cx1, cy1, cx2, cy2 = centroOlho(olho_direito, olho_direito_pb, olho_esquerdo, olho_esquerdo_pb)

        # IDENTIFICAÇÃO DA IRIS
        vetor_olho_direito, vetor_olho_esquerdo = identificacaoIris(olho_direito_cinza, olho_esquerdo_cinza)

        # MOSTRA IDENTIFICAÇÕES DA IRIS DO OLHO DIREITO
        cx3, cy3, ind1, raio1 = showIrisDireita(vetor_olho_direito, olho_direito_cinza)

        # MOSTRA IDENTIFICAÇÕES DA IRIS DO OLHO ESQUERDO
        cx4, cy4, ind2, raio2 = showIrisEsquerda(vetor_olho_esquerdo, olho_esquerdo_cinza)

        # MOSTRA O RESULTADO
        resultado(raio1, ind1, cx3, cx1, cy3, cy1, raio2, ind2, cx4, cx2, cy4, cy2)


def preProcessamento(imagem):
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    return imagem_cinza


def identificacaoDlib(imagem_cinza):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    for rosto in detector(imagem_cinza):
        pontos = predictor(imagem_cinza, rosto)
    return detector(imagem_cinza), pontos


def pontoOlhos(pontos_dlib, pontos):
    for n in range(0, 68):
        x = pontos.part(n).x
        y = pontos.part(n).y
        cv2.circle(pontos_dlib, (x, y), 6, (255, 0, 0), -1)

    # st.text("Pontos da biblioteca dlib")
    # st.image(pontos_dlib)

    olho_direito_pontos = np.array([(pontos.part(36).x, pontos.part(36).y),
                                    (pontos.part(37).x, pontos.part(37).y),
                                    (pontos.part(38).x, pontos.part(38).y),
                                    (pontos.part(39).x, pontos.part(39).y),
                                    (pontos.part(40).x, pontos.part(40).y),
                                    (pontos.part(41).x, pontos.part(41).y)], np.int32)

    olho_esquerdo_pontos = np.array([(pontos.part(42).x, pontos.part(42).y),
                                     (pontos.part(43).x, pontos.part(43).y),
                                     (pontos.part(44).x, pontos.part(44).y),
                                     (pontos.part(45).x, pontos.part(45).y),
                                     (pontos.part(46).x, pontos.part(46).y),
                                     (pontos.part(47).x, pontos.part(47).y)], np.int32)

    return olho_direito_pontos, olho_esquerdo_pontos


def separacaoOlhos(imagem, olho_direito_pontos, olho_esquerdo_pontos):
    # CRIAÇÃO DE UMA MÁSCARA DA IMAGEM
    height, width, _ = imagem.shape  # DIMENSÕES
    mask = np.zeros((height, width), np.uint8)

    # "LIGA" OS PIXELS DA REGIÃO DOS OLHOS
    cv2.polylines(mask, [olho_direito_pontos], True, 255, 2)
    cv2.fillPoly(mask, [olho_direito_pontos], 255)
    cv2.polylines(mask, [olho_esquerdo_pontos], True, 255, 2)
    cv2.fillPoly(mask, [olho_esquerdo_pontos], 255)
    olhos_mask = cv2.bitwise_and(imagem, imagem, mask=mask)
    # st.image(olhos_mask)

    # SEPARAÇÃO DOS DOIS OLHOS
    min_x_d = np.min(olho_direito_pontos[:, 0])
    max_x_d = np.max(olho_direito_pontos[:, 0])
    min_y_d = np.min(olho_direito_pontos[:, 1])
    max_y_d = np.max(olho_direito_pontos[:, 1])
    olho_direito = olhos_mask[min_y_d: max_y_d, min_x_d: max_x_d]
    olho_direito_cinza = cv2.cvtColor(olho_direito, cv2.COLOR_BGR2GRAY)

    min_x_e = np.min(olho_esquerdo_pontos[:, 0])
    max_x_e = np.max(olho_esquerdo_pontos[:, 0])
    min_y_e = np.min(olho_esquerdo_pontos[:, 1])
    max_y_e = np.max(olho_esquerdo_pontos[:, 1])
    olho_esquerdo = olhos_mask[min_y_e: max_y_e, min_x_e: max_x_e]
    olho_esquerdo_cinza = cv2.cvtColor(olho_esquerdo, cv2.COLOR_BGR2GRAY)

    #c1, c2 = st.columns(2)
    # c1.markdown('***Olho direito***')
    #c1.image(olho_direito)

    #c2.markdown('***Olho esquerdo***')
    #c2.image(olho_esquerdo)
    # st.markdown("")

    st.subheader('***IDENTIFICAÇÃO DO CENTRO DO OLHO***')
    st.markdown('Aumente o valor da barra até que o ponto vermelho esteja no mesmo ponto do reflexo da luz.')
    st.markdown('***Exemplo:***')
    exemplo_thre = cv2.imread('exemplo Threshold.jpeg')
    st.image(exemplo_thre)
    st.markdown('')

    return olho_direito, olho_direito_cinza, olho_esquerdo, olho_esquerdo_cinza


def thresholding(olho_direito_cinza, olho_esquerdo_cinza):
    c1, c2 = st.columns(2)

    thresholdValue1 = c1.slider('REFLEXO DO OLHO DIREITO', 0, 255)  # SLIDE BAR PARA O THRESHOLD
    _, olho_direito_pb = cv2.threshold(olho_direito_cinza, thresholdValue1, 255, cv2.THRESH_BINARY)

    thresholdValue2 = c2.slider('REFLEXO DO OLHO ESQUERDO', 0, 255)  # SLIDE BAR PARA O THRESHOLD
    _, olho_esquerdo_pb = cv2.threshold(olho_esquerdo_cinza, thresholdValue2, 255, cv2.THRESH_BINARY)

    #c1.image(olho_direito_pb)
    #c2.image(olho_esquerdo_pb)

    return olho_direito_pb, olho_esquerdo_pb


def centroOlho(olho_direito, olho_direito_pb, olho_esquerdo, olho_esquerdo_pb):
    c1, c2 = st.columns(2)

    centro_direito = olho_direito.copy()
    height, width, _ = centro_direito.shape

    contours1, hierarchy = cv2.findContours(olho_direito_pb, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)

    mx = []
    my = []
    mex = 0
    mey = 0
    for contours in contours1:
        for contour in contours:
            for contou in contour:
                x = contou[0]
                y = contou[1]
                mex += x
                mey += y
        mex = int(mex / len(contours))
        mey = int(mey / len(contours))
        mx.append(mex)
        my.append(mey)
        mex = 0
        mey = 0

    for i in range(0, len(mx)):
        x = mx[i]
        y = my[i]
        if width * 2 / 3 >= x >= width * 1 / 3:
            cx1 = x
            cy1 = y

    cv2.circle(centro_direito, (cx1, cy1), 2, (255, 0, 0), -1)

    centro_esquerdo = olho_esquerdo.copy()
    height, width, _ = centro_esquerdo.shape
    contours2, hierarchy = cv2.findContours(olho_esquerdo_pb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mx = []
    my = []
    mex = 0
    mey = 0

    for contours in contours2:
        for contour in contours:
            for contou in contour:
                x = contou[0]
                y = contou[1]
                mex += x
                mey += y
        mex = int(mex / len(contours))
        mey = int(mey / len(contours))
        mx.append(mex)
        my.append(mey)
        mex = 0
        mey = 0

    for i in range(0, len(mx)):
        x = mx[i]
        y = my[i]
        if width * 2 / 3 >= x >= width * 1 / 3:
            cx2 = x
            cy2 = y

    cv2.circle(centro_esquerdo, (cx2, cy2), 2, (255, 0, 0), -1)

    c1.markdown("***Centro do olho direito:***")
    c1.image(centro_direito)
    c2.markdown("***Centro do olho esquerdo:***")
    c2.image(centro_esquerdo)
    st.markdown("***")

    return cx1, cy1, cx2, cy2


def identificacaoIris(olho_direito_cinza, olho_esquerdo_cinza):
    olho_direito_blur = cv2.medianBlur(olho_direito_cinza, 5)  # APLICAÇÃO DO BLUR
    olho_esquerdo_blur = cv2.medianBlur(olho_esquerdo_cinza, 5)
    #st.image(olho_direito_blur)
    #st.image(olho_direito_blur)
    height1, width1 = olho_direito_blur.shape
    height2, width2 = olho_esquerdo_blur.shape
    raio_min1 = int(height1 / 2)
    raio_min2 = int(height2 / 2)
    raio_max1 = height1
    raio_max2 = height2
    dist_min1 = width1
    dist_min2 = width2
    dp = 1.0

    # VETOR QUE IRA ARMAZENAR OS RESULTADOS DOS CIRCULOS IDENTIFICADOS:
    vetor_olho_direito = []
    vetor_olho_esquerdo = []



    # LAÇO DE IDENTIFICAÇÃO
    while dp <= 2.10:
        circles1 = cv2.HoughCircles(olho_direito_blur, cv2.HOUGH_GRADIENT, dp, dist_min1, param1=50, param2=30,
                                    minRadius=raio_min1, maxRadius=raio_max1)
        circles2 = cv2.HoughCircles(olho_esquerdo_blur, cv2.HOUGH_GRADIENT, dp, dist_min2, param1=50, param2=30,
                                    minRadius=raio_min2, maxRadius=raio_max2)

        if np.all(circles1) == None:
            pass
        else:
            circles1 = np.uint16(np.around(circles1))
        if np.all(circles2) == None:
            pass
        else:
            circles2 = np.uint16(np.around(circles2))

        vetor_olho_direito.append(circles1)
        vetor_olho_esquerdo.append(circles2)
        dp = dp + 0.10
    return vetor_olho_direito, vetor_olho_esquerdo


def showIrisDireita(vetor_olho_direito, olho_direito_cinza):
    vetor_iris_direita = []
    cx3 = []
    cy3 = []
    raio = []

    for i in range(0, len(vetor_olho_direito)):
        iris_d = olho_direito_cinza.copy()
        if np.all(vetor_olho_direito[i]) == None:
            pass
        else:
            for (x, y, r) in vetor_olho_direito[i][0][0:]:
                cv2.circle(iris_d, (x, y), r, 255, 1)
                cv2.circle(iris_d, (x, y), 2, 255, 1)

                cx3.append(x)
                cy3.append(y)
                raio.append(r)
                vetor_iris_direita.append(iris_d)

    st.subheader('***IRIS DIREITA***')
    st.markdown('Usando a barra, selecione a imagem em que o círculo melhor se encaixa no limbo (borda da íris):')
    st.markdown('(Caso as imagens sejam semelhantes, escolha a de sua preferência)')
    st.markdown('***Exemplo:***')
    exemplo = cv2.imread('exemplo Limbo.jpeg')
    st.image(exemplo)
    st.header('')
    c1, c2, c3 = st.columns(3)

    c = 1
    while c <= 3:
        for i in range(0, len(vetor_iris_direita)):
            if c == 1:
                with c1:
                    st.write("Imagem: ", i + 1)
                    c1.image(vetor_iris_direita[i])
            elif c == 2:
                with c2:
                    st.write("Imagem: ", i + 1)
                    c2.image(vetor_iris_direita[i])
            elif c == 3:
                with c3:
                    st.write("Imagem: ", i + 1)
                    c3.image(vetor_iris_direita[i])
                    c = 0
            c += 1
        break

    if len(vetor_iris_direita) > 1:
        indice1 = st.slider('SELEÇÃO DA IMAGEM:', 1, len(vetor_iris_direita))
        st.write('Imagem selecionada:', indice1)
        ind1 = indice1 - 1
        st.image(vetor_iris_direita[ind1])
        st.markdown('***')
    return cx3, cy3, ind1, raio


def showIrisEsquerda(vetor_olho_esquerdo, olho_esquerdo_cinza):
    vetor_iris_esquerda = []
    cx4 = []
    cy4 = []
    raio = []

    for i in range(0, len(vetor_olho_esquerdo)):
        iris_e = olho_esquerdo_cinza.copy()
        if np.all(vetor_olho_esquerdo[i]) == None:
            pass
        else:
            for (x, y, r) in vetor_olho_esquerdo[i][0][0:]:
                cx4.append(x)
                cy4.append(y)
                raio.append(r)

                # draw the outer circle
                cv2.circle(iris_e, (x, y), r, 255, 1)
                # draw the center of the circle
                cv2.circle(iris_e, (x, y), 2, 255, 1)
                vetor_iris_esquerda.append(iris_e)

    st.subheader('***IRIS ESQUERDA***')
    st.markdown(
        'Como na etapa anterior, usando a barra, selecione a imagem em que o círculo melhor se encaixa no limbo (borda da íris):')
    st.markdown('(Caso as imagens sejam semelhantes, escolha a de sua preferência)')
    c1, c2, c3 = st.columns(3)
    c = 1
    while c <= 3:
        for i in range(0, len(vetor_iris_esquerda)):
            if c == 1:
                with c1:
                    st.write("Imagem: ", i + 1)
                    c1.image(vetor_iris_esquerda[i])
            elif c == 2:
                with c2:
                    st.write("Imagem: ", i + 1)
                    c2.image(vetor_iris_esquerda[i])
            elif c == 3:
                with c3:
                    st.write("Imagem: ", i + 1)
                    c3.image(vetor_iris_esquerda[i])
                    c = 0
            c += 1
        break

    if len(vetor_iris_esquerda) > 1:
        indice2 = st.slider('SELEÇÃO DA IMAGEM: ', 1, len(vetor_iris_esquerda))
        st.write('Imagem selecionada:', indice2)
        ind2 = indice2 - 1
        st.image(vetor_iris_esquerda[ind2])
        st.markdown("***")
    return cx4, cy4, ind2, raio


def resultado(raio1: object, ind1, cx3, cx1, cy3, cy1, raio2, ind2, cx4, cx2, cy4, cy2):
    st.header("RESULTADO")

    distancia_x_direita = abs(cx3[ind1] - cx1)  # DISTANCIA DO REFLEXO ATE O CENTRO DO LIMBO
    distancia_y_direita = abs(cy3[ind1] - cy1)  # DISTANCIA DO REFLEXO ATE O CENTRO DO LIMBO
    distancia_direita = sqrt(distancia_x_direita * distancia_x_direita + distancia_y_direita * distancia_y_direita)

    distancia_pupila = raio1[ind1] * 0.15  # pupila
    distancia_metade_iris = raio1[ind1] * 0.30  # metade iris

    if distancia_direita <= distancia_pupila:
        st.text('Baixa probabilidade de existência de estrabismo no olho direito.')
    if distancia_metade_iris >= distancia_direita > distancia_pupila:
        st.text('Probabilidade de existência de estrabismo no olho direito.')
    if distancia_direita > distancia_metade_iris:
        st.text('Alta probabilidade de existência de estrabismo no olho direito.')

    distancia_x_esquerda = abs(cx4[ind2] - cx2)
    distancia_y_esquerda = abs(cy4[ind2] - cy2)
    distancia_esquerda = sqrt(distancia_x_esquerda * distancia_x_esquerda + distancia_y_esquerda * distancia_y_esquerda)

    distancia_pupila2 = raio2[ind2] * 0.15  # pupila
    distancia_metade_iris2 = raio2[ind2] * 0.30  # metade iris

    if distancia_esquerda <= distancia_pupila2:
        st.text('Baixa probabilidade de existência de estrabismo no olho esquerdo.')
    if distancia_metade_iris2 >= distancia_esquerda > distancia_pupila2:
        st.text('Probabilidade de existência de estrabismo no olho esquerdo.')
    if distancia_esquerda > distancia_metade_iris2:
        st.text('Alta probabilidade de existência de estrabismo no olho esquerdo.')

    st.markdown('***')
    st.markdown(
        '***Independente do resultado é aconselhado consultar um profissional para realizar um exame oftamológico completo, '
        'pois apenas o oftalmologista poderá fazer o diagnóstico e realizar o devido tratamento.***')
    st.markdown('***')
    return


if __name__ == '__main__':
    main()
