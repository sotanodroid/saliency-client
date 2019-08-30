module.exports = {
	title: 'Saliency.ai documentation',
	theme: 'saliency-theme',
	description: 'We enable you to use AI in your R&D',
	themeConfig: {
		logo: '/light-bg.png',
		repo: 'saliency-ai/saliency-client',
		nav: [
			{ text: 'Home', link: '/' },
			{ text: 'Guide', link: '/guide/' },
			/*      { text: 'GitHub', link: 'https://github.com/saliency-ai/saliency-client/' },*/
		],
		sidebar: {
			'/guide/': [
				'',     
			],/*
			'/': [
				'',     
				'one',  
				'two'   
			]*/
		}
	}
}
