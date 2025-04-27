# class ResNetEncoder(nn.Module):
#     def __init__(self,
#                  n_ResidualBlock=8,
#                  n_levels=4,
#                  z_dim=10,
#                  bUseMultiResSkips=False):

#         super(ResNetEncoder, self).__init__()

#         self.max_filters = 2 ** (n_levels+3)
#         self.n_levels = n_levels
#         self.bUseMultiResSkips = bUseMultiResSkips

#         self.conv_list = []
#         self.res_blk_list = []
#         self.multi_res_skip_list = []
        
#         self.inplanes = 8
        
#         self.input_conv = nn.Sequential(
#             nn.Conv2d(1,self.inplanes, kernel_size=3,
#                                    stride=1, padding='same'),
#             nn.BatchNorm2d(self.inplanes),
#             nn.ReLU()
#         )

#         for i in range(n_levels):
#             n_filters_1 = min(2 ** (i + 3), z_dim)
#             n_filters_2 = min(2 ** (i + 4), z_dim)
#             ks = 2 ** (n_levels - i)
            
#             layers = []
#             for _ in range(n_ResidualBlock):
#                 layers.append(ResidualBlock(n_filters_1,n_filters_1))

#             self.res_blk_list.append(
#                 nn.Sequential(*layers)
#             )

#             self.conv_list.append(
#                 nn.Sequential(
#                     nn.Conv2d(n_filters_1,n_filters_2, kernel_size=2,
#                                            stride=2),
#                     nn.BatchNorm2d(n_filters_2),
#                     nn.ReLU(inplace=True),
#                 )
#             )

#             # if bUseMultiResSkips:
#             #     self.multi_res_skip_list.append(
#             #         nn.Sequential([
#             #             nn.Conv2d(self.max_filters,self.max_filters, kernel_size=(ks, ks),
#             #                                    strides=(ks, ks), padding='same'),
#             #             nn.BatchNormalization(),
#             #             nn.LeakyReLU(alpha=0.2),
#             #         ])
#             #     )
#         #self.output_conv = nn.AvgPool2d((2, 2))
#         self.output_conv = nn.Conv2d(z_dim,z_dim, kernel_size=3,
#                                                   stride=1, padding='same')

#     def forward(self, x):

#         print(f'Encoder: input_conv {self.input_conv}')

#         for i in range(self.n_levels):
#             print(f'res_blk_list {i} {self.res_blk_list[i]}')

#             print(f'conv_list {i} {self.conv_list[i]}')

#         print(f'output_conv: {self.output_conv}') 

#         print('---------------')
#         x = self.input_conv(x)
       

#         skips = []
#         for i in range(self.n_levels):
            
#             x = self.res_blk_list[i](x)
#             # if self.bUseMultiResSkips:
#             #     skips.append(self.multi_res_skip_list[i](x))
#             x = self.conv_list[i](x)

#         # if self.bUseMultiResSkips:
#         #     x = sum([x] + skips)

#         x = self.output_conv(x)

#         return x


# class ResNetDecoder(nn.Module):
#     def __init__(self,
#                  n_ResidualBlock=8,
#                  n_levels=4,z_dim=128,
#                  output_channels=3,
#                  bUseMultiResSkips=False ):

#         super(ResNetDecoder, self).__init__()

#         self.max_filters = min(2 ** (n_levels+3),z_dim)
#         self.n_levels = n_levels
#         self.bUseMultiResSkips = bUseMultiResSkips

#         self.conv_list = []
#         self.res_blk_list = []
#         self.multi_res_skip_list = []

#         self.input_conv = nn.Sequential(
#             nn.Conv2d(self.max_filters,self.max_filters, kernel_size=3,
#                                    stride=1, padding='same'),
#             nn.BatchNorm2d(self.max_filters),
#             nn.ReLU(inplace=True),
#         )

#         for i in range(n_levels):
#             n_filters = 2 ** (self.n_levels - i + 2)
#             ks = 2 ** (i + 1)

#             layers = []
#             for _ in range(n_ResidualBlock):
#                 layers.append(ResidualBlock(n_filters,n_filters))
#             self.res_blk_list.append(
#                 nn.Sequential(*layers)
#             )



#             self.conv_list.append(
#                 nn.Sequential(
#                     nn.ConvTranspose2d(n_filters,n_filters, kernel_size=2,
#                                                     stride=2),
#                     nn.BatchNorm2d(n_filters),
#                     nn.ReLU(inplace=True),
#                 )
#             )

#             # if bUseMultiResSkips:
#             #     self.multi_res_skip_list.append(
#             #         nn.Sequential(
#             #             nn.ConvTranspose2d(n_filters,n_filters, kernel_size=ks,
#             #                                             stride=ks),
#             #             nn.BatchNorm2d(n_filters),
#             #             nn.ReLU(inplace=True),
#             #         )
#             #     )

#         self.output_conv = nn.Conv2d(output_channels,output_channels, kernel_size=3,
#                                                   stride=1, padding='same')

#     def forward(self, z):

#         z = z_top = self.input_conv(z)

#         for i in range(self.n_levels):
#             z = self.conv_list[i](z)
#             z = self.res_blk_list[i](z)
#             if self.bUseMultiResSkips:
#                 z += self.multi_res_skip_list[i](z_top)

#         z = self.output_conv(z)

#         return z


class ResNetAE(nn.Module):
    def __init__(self,
                 input_shape=(256, 256, 1),
                 n_ResidualBlock=8,
                 n_levels=4,
                 z_dim=128,
                 bottleneck_dim=128,
                 bUseMultiResSkips=True):
        super(ResNetAE, self).__init__()

        self.batchSize = 8
        assert input_shape[0] == input_shape[1]
        output_channels = input_shape[2]
        self.z_dim = z_dim
        self.img_latent_dim = input_shape[0] // (2 ** n_levels)

        self.encoder = ResNetEncoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels,
                                     z_dim=z_dim, bUseMultiResSkips=bUseMultiResSkips)
        self.decoder = ResNetDecoder(n_ResidualBlock=n_ResidualBlock, n_levels=n_levels, z_dim=z_dim,
                                     output_channels=input_shape[0], bUseMultiResSkips=bUseMultiResSkips)

        self.fc1 = nn.Linear(self.img_latent_dim * self.img_latent_dim*bottleneck_dim,output_channels)
        self.fc2 = nn.Linear(output_channels,self.img_latent_dim * self.img_latent_dim * self.z_dim)

    def encode(self, x):
        h = self.encoder(x)
        print(f'img_latent_dim: {self.img_latent_dim}, z_dim: {self.z_dim}')
        print(f'original h: {h.shape}')
        #h = torch.flatten(h, 1)
        
        h = torch.reshape(h, (-1, self.batchSize, self.img_latent_dim * self.img_latent_dim * self.z_dim))
        print(f'h after reshapre: {h.shape}')
        return self.fc1(h),h

    def decode(self, z):
        print(f'original z: {z.shape}')
        z = self.fc2(z)

        #z = torch.reshape(z, (1, self.img_latent_dim* self.img_latent_dim * self.z_dim))
        #print(f'after reshape z: {z.shape}')
        
       
        h = self.decoder(z)
        return nn.sigmoid(h)

    def forward(self, x):
        y_logits, embedding = self.encode(x)
        print(f'y_logits: {y_logits.shape}')
        print(f'embedding: {embedding.shape}')
        
        return self.decode(embedding)
        #return self.decode(self.encode(x))



