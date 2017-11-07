#feature model	
	def fcrnn(self):
		input_layer = Input(shape=self.input_shape)
		x = TimeDistributed(Flatten())(input_layer)
		x = LSTM(256, return_sequences=True)(x)
		x = LSTM(256, return_sequences=True)(x)
		x = LSTM(256, return_sequences=True)(x)
		x = LSTM(256, return_sequences=True)(x)
		x = LSTM(256)(x)
		#x = Flatten()(x)
		# x = Dropout(0.5)(x)
		# x = Dense(512, activation='relu')(x)
		x = Dropout(0.5)(x)
		x = Dense(self.nb_classes, activation='softmax')(x)
		model = Model(inputs=input_layer, outputs=x)
		print(model.summary())

		return model

	def crnn1(self):
		model = Sequential()
		model.add(GRU(256, input_dim=35840))
		model.add(Dropout(0.5))
		model.add(Dense(self.nb_classes, activation='softmax'))
		print(model.summary())

		return model

	def mcnn1(self):
		input_layer = Input(shape=self.input_shape)
		x = TimeDistributed(Flatten())(input_layer)
		x = MaxPooling1D(pool_size=self.input_shape[0])(x)
		x = Dropout(0.5)(x)
		x = Dense(512, activation='relu')(x)
		x = Flatten()(x)
		x = Dropout(0.5)(x)
		x = Dense(self.nb_classes, activation='softmax')(x)
		model = Model(inputs=input_layer, outputs=x)
		print(model.summary())

		return model

	def mcnnall(self):
		input_layer = Input(shape=[35840])
		x = Dropout(0.5)(input_layer)
		x = Dense(512, activation='relu')(x)
		#x = Flatten()(x)
		x = Dropout(0.5)(x)
		x = Dense(self.nb_classes, activation='softmax')(x)
		model = Model(inputs=input_layer, outputs=x)
		print(model.summary())

		return model

	def mcnn3(self):
		input_layer = Input(shape=self.input_shape)
		x = TimeDistributed(Flatten())(input_layer)
		x = TimeDistributed(Dropout(0.5))(x)
		x = TimeDistributed(Dense(512, activation='relu'))(x)
		x = TimeDistributed(Dropout(0.5))(x)
		x = TimeDistributed(Dense(self.nb_classes))(x)
		x = MaxPooling1D(pool_size=self.input_shape[0])(x)
		x = Flatten()(x)
		x = Activation('softmax')(x)
		model = Model(inputs=input_layer, outputs=x)
		print(model.summary())

		return model

	def mcnn2(self):
		input_layer = Input(shape=self.input_shape)
		x = TimeDistributed(Flatten())(input_layer)
		x = TimeDistributed(Dropout(0.5))(x)
		x = TimeDistributed(Dense(512, activation='relu'))(x)
		x = AveragePooling1D(pool_size=self.input_shape[0])(x)
		x = Flatten()(x)
		x = Dropout(0.5)(x)
		x = Dense(self.nb_classes, activation='softmax')(x)
		model = Model(inputs=input_layer, outputs=x)
		print(model.summary())
		#def selu(x):
		# alpha = 1.6732632423543772848170429916717
		# lamb = 1.0507009873554804934193349852946
		# return lamb * np.where(x > 0., x, alpha * np.exp(x) - alpha)

		return model