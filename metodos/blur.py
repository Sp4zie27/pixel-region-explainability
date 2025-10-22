import cv2

# Carrega a imagem
imagem = cv2.imread("Imagens_teste/dogs/5861.jpg")

# Aplica um blur leve (5x5 kernel)
imagem_blur = cv2.GaussianBlur(imagem, (15, 15), 0)

# Salva a imagem resultante
cv2.imwrite("Imagens_teste/dogs/imagem_blur.jpg", imagem_blur)

# Mostra a imagem original e a borrada (opcional)
cv2.imshow("Original", imagem)
cv2.imshow("Blur leve", imagem_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
