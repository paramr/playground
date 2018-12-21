import React, { Component } from 'react'
import { SheetsRegistry } from 'jss';
import { create } from 'jss';
import JssProvider from 'react-jss/lib/JssProvider'
import { MuiThemeProvider } from '@material-ui/core/styles'
import { createMuiTheme } from '@material-ui/core/styles'
import { createGenerateClassName } from '@material-ui/core/styles'

import theme from './src/theme'

export default {
  getSiteData: () => ({
    title: 'React Static',
  }),
  getRoutes: async () => {
    return []
  },
  renderToElement: async (App, { meta, clientStats }) => {
    // Create a sheetsRegistry instance.
    const sheetsRegistry = new SheetsRegistry()

    // Create a MUI theme instance.
    const muiTheme = createMuiTheme(theme)

    const generateClassName = createGenerateClassName()

    meta.jssStyles = sheetsRegistry.toString()

    return (
      <JssProvider registry={sheetsRegistry} generateClassName={generateClassName}>
        <MuiThemeProvider theme={muiTheme} sheetsManager={new Map()}>
          <App />
        </MuiThemeProvider>
      </JssProvider>
    )
  },
  Document: class CustomHtml extends Component {
    render () {
      const {
        Html, Head, Body, children, renderMeta,
      } = this.props
      return (
        <Html>
          <Head>
            <meta charSet="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <link
              href="https://fonts.googleapis.com/css?family=Roboto:300,400,500"
              rel="stylesheet"
            />
          </Head>
          <Body>
            {children}
            <style id="jss-server-side">{renderMeta.jssStyles}</style>
          </Body>
        </Html>
      )
    }
  },
}
